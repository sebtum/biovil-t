# inference.py
from __future__ import annotations
import argparse
import logging
from typing import List, Dict
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import math, time
# --- add near top of your file ---
import json, os, pathlib, numpy as np
from contextlib import contextmanager

from biovilt_model import BioviltPairEncoder, BioviltText, build_prompt_bank, score_margins_pairwise, decide
from pair_cxr_dataset_lite import PairCXRDatasetLite  # your dataset

log = logging.getLogger("biovilt.infer")

def setup_logging(verbosity: int = 1):
    level = logging.INFO if verbosity >= 1 else logging.WARNING
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s: %(message)s")

def infer(args):
    setup_logging(args.v)
    log.info("Loading CSV: %s", args.csv)
    wrote_header = False

    def flush_recs():
        nonlocal wrote_header, recs
        if not recs:
            return
        df_chunk = pd.DataFrame(recs)
        mode = "a" if wrote_header else "w"
        df_chunk.to_csv(args.out, index=False, mode=mode, header=not wrote_header)
        log.info(f"Wrote {len(recs)} records to {args.out} (mode={mode})")
        wrote_header = True
        recs.clear()

    skipped = 0
    processed = 0
    
    ds = PairCXRDatasetLite(args.csv, bad_dir=args.bad_dir)
    # Ensure ids exist (depends on your class)
    if not hasattr(ds, "ids") or len(getattr(ds, "ids")) == 0:
        ds.ids = list(ds.df.index.astype(str))

    total_samples = len(ds)
    log.info("Dataset size: %d samples", total_samples)
    total_batches = math.ceil(len(ds) / args.batch_size) if len(ds) else 0
    log.info("Total batches: %d (batch_size=%d)", total_batches, args.batch_size)
    t0 = time.time()

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # Models + caches
    pair_enc = BioviltPairEncoder(device=device, half=args.half, enable_cache=True)
    txt = BioviltText(device=device)

    # Use diseases/columns directly from dataset (already computed in your class)
    diseases: List[str] = ds.diseases
    add_cols: List[str] = ds.add_cols
    miss_cols: List[str] = ds.miss_cols

    log.info("Diseases: %s", ", ".join(diseases))

    # Build & cache text prompt banks once
    log.info("Building text prompt banks...")
    banks: Dict[str, Dict[str, torch.Tensor]] = {d: build_prompt_bank(txt, d) for d in diseases}
    log.info("Text prompts cached: %d banks", len(banks))

    # --- init embedding writer lazily after first successful forward to know D ---
    emb_writer = None

    recs = []

    # get 1 sample
    sample = ds[0]
    with torch.inference_mode():
        e = pair_enc.encode_pair(
            sample["pixel_values_curr"].unsqueeze(0),
            sample["pixel_values_prev"].unsqueeze(0),
            dicom_id=[sample["dicom_id"]],
            ref_dicom_id=[sample["ref_dicom_id"]]
        )

    emb_dim = e.shape[1]
    emb_writer = EmbeddingWriter(
        root=args.emb_dir,
        total_capacity=len(ds),
        emb_dim=emb_dim,
        dtype=args.emb_dtype,
        resume=True #args.resume
    )
    log.info("Embedding sink: %s (capacity=%d, dim=%d, dtype=%s, start_ptr=%d)",
                         str(emb_writer.emb_path), emb_writer.total_capacity, emb_writer.emb_dim,
                         args.emb_dtype, emb_writer.filled())
    start_idx = emb_writer.filled()      # number of pairs already written
    if start_idx > 0:
        log.info("Resuming from index %d", start_idx)

    # simple skip in the main loop:
    seen = 0
    chunk_size = 1024
    log.info("Starting inference...")
    with torch.inference_mode():
        for b_idx, batch in enumerate(loader, start=1):
            if batch is None:        # whole batch filtered out
                skipped += args.batch_size
                continue
            Blen = len(batch["dicom_id"])
            if seen + Blen <= start_idx:
                seen += Blen
                continue
            # if partially overlapping batch, slice:
            if seen < start_idx:
                off = start_idx - seen
                for k in batch:
                    batch[k] = batch[k][off:]
                Blen = len(batch["dicom_id"])
                seen = start_idx
            x_curr = batch["pixel_values_curr"]
            x_prev = batch["pixel_values_prev"]
            dicom_ids = batch["dicom_id"]
            ref_ids   = batch["ref_dicom_id"]

            try:
                # ... your forward pass / scoring ...
                # Pair embeddings with caching by (dicom_id, ref_dicom_id)
                e = pair_enc.encode_pair(x_curr, x_prev, dicom_id=dicom_ids, ref_dicom_id=ref_ids)  # [B,D]
                processed += len(next(iter(batch.values()))) if isinstance(batch, dict) else len(batch)
            except Exception as ex:
                log.exception("Model failure at step=%d: %s", b_idx, ex)
                continue  # keep going

            # write embeddings + ids
            try:
                emb_writer.add_batch(e, dicom_ids, ref_ids)
            except Exception as ex:
                log.exception("Failed to write embeddings at batch=%d: %s", b_idx, ex)
            # Prepare row dict for this mini-batch
            row = {
                "dicom_id": dicom_ids,
                "ref_dicom_id": ref_ids,
            }

            for d in diseases:
                s_add, s_miss = score_margins_pairwise(e, banks[d])  # [B], [B]
                lab = decide(s_add, s_miss, args.tau_add, args.tau_miss, args.delta_tie)

                row.setdefault(f"pred_additional_{d}", []).extend([1 if l == "additional" else 0 for l in lab])
                row.setdefault(f"pred_missing_{d}",   []).extend([1 if l == "missing"    else 0 for l in lab])
                row.setdefault(f"margin_add_{d}", []).extend([float(x) for x in s_add.tolist()])
                row.setdefault(f"margin_miss_{d}",[]).extend([float(x) for x in s_miss.tolist()])

                # GT from dataset labels (always returned)
                # Map disease index for y_add/y_miss vectors
                d_idx = diseases.index(d)
                if "y_add" in batch:
                    row.setdefault(f"gt_additional_{d}", []).extend([int(v) for v in batch["y_add"][:, d_idx].tolist()])
                if "y_miss" in batch:
                    row.setdefault(f"gt_missing_{d}", []).extend([int(v) for v in batch["y_miss"][:, d_idx].tolist()])

            # fan out
            B = len(dicom_ids)
            for i in range(B):
                rec = {"dicom_id": dicom_ids[i], "ref_dicom_id": ref_ids[i]}
                for d in diseases:
                    rec[f"pred_additional_{d}"] = row[f"pred_additional_{d}"][i]
                    rec[f"pred_missing_{d}"]    = row[f"pred_missing_{d}"][i]
                    rec[f"margin_add_{d}"]      = row[f"margin_add_{d}"][i]
                    rec[f"margin_miss_{d}"]     = row[f"margin_miss_{d}"][i]
                    if f"gt_additional_{d}" in row:
                        rec[f"gt_additional_{d}"] = row[f"gt_additional_{d}"][i]
                    if f"gt_missing_{d}" in row:
                        rec[f"gt_missing_{d}"] = row[f"gt_missing_{d}"][i]
                recs.append(rec)
            if (b_idx % args.log_interval == 0) or (b_idx == total_batches):
                elapsed = time.time() - t0
                ips = (b_idx * args.batch_size) / max(elapsed, 1e-6)  # images/sec (approx)
                log.info("Progress: batch %d/%d (%.1f%%), elapsed=%.1fs, approx %.1f imgs/s, cache hits=%d, misses=%d, skipped=%d",
                         b_idx, total_batches, 100.0 * b_idx / max(total_batches, 1),
                         elapsed, ips, pair_enc.cache_hits, pair_enc.cache_misses, skipped)
            # periodic flush + ckpt update
            if emb_writer.filled() % args.flush_every == 0:
                emb_writer.flush()
            if len(recs) >= chunk_size:
                flush_recs()
    emb_writer.flush()
    emb_writer.close()
    if recs:
        flush_recs()
    log.info("Pair cache hits=%d, misses=%d", pair_enc.cache_hits, pair_enc.cache_misses)
    log.info("Done. processed=%d skipped~=%d. Bad samples written to %s",
             processed, skipped, args.bad_dir or "<none>")



class EmbeddingWriter:
    """
    Progressive, resumable writer for pair embeddings + id mapping.
    - embeddings: single memmap .npy file with fixed shape [N, D]
    - ids: append-only Parquet (or CSV fallback) containing dicom_id/ref_dicom_id in the same row order
    - checkpoint.json tracks write_ptr and config for resume
    """
    def __init__(self, root: str, total_capacity: int, emb_dim: int, dtype: str = "float16", resume: bool = True):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.root / "pairs.mmapped.npy"
        self.ckpt_path = self.root / "checkpoint.json"
        self.ids_parquet = self.root / "ids.parquet"
        self.ids_csv = self.root / "ids.csv"  # fallback if pyarrow is missing
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.total_capacity = int(total_capacity)
        self.emb_dim = int(emb_dim)

        self.write_ptr = 0
        self._init_or_resume(resume)

    def _init_or_resume(self, resume: bool):
        if resume and self.ckpt_path.exists() and self.emb_path.exists():
            with open(self.ckpt_path, "r") as f:
                ck = json.load(f)
            # sanity checks
            if (ck.get("total_capacity") == self.total_capacity and
                ck.get("emb_dim") == self.emb_dim and
                ck.get("dtype") == str(self.dtype).split("'")[1]):
                self.write_ptr = int(ck.get("write_ptr", 0))
                self.mm = np.memmap(self.emb_path, dtype=self.dtype, mode="r+",
                                    shape=(self.total_capacity, self.emb_dim))
                return
            else:
                # mismatch -> start fresh (safe choice)
                pass

        # fresh create
        self.mm = np.memmap(self.emb_path, dtype=self.dtype, mode="w+",
                            shape=(self.total_capacity, self.emb_dim))
        self.write_ptr = 0
        self._save_ckpt()

        # remove old ids if present (we're fresh)
        if self.ids_parquet.exists():
            self.ids_parquet.unlink()
        if self.ids_csv.exists():
            self.ids_csv.unlink()

    def _save_ckpt(self):
        tmp = self.ckpt_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({
                "write_ptr": self.write_ptr,
                "total_capacity": self.total_capacity,
                "emb_dim": self.emb_dim,
                "dtype": str(self.dtype).split("'")[1],
                "emb_path": str(self.emb_path),
                "ids_sink": "parquet" if self.ids_parquet.exists() else ("csv" if self.ids_csv.exists() else "none")
            }, f)
        os.replace(tmp, self.ckpt_path)

    def add_batch(self, emb: torch.Tensor, dicom_ids: List[str], ref_ids: List[str]):
        emb = emb.detach().to("cpu")
        B, D = emb.shape
        assert D == self.emb_dim, f"Embedding dim changed: got {D}, expected {self.emb_dim}"
        end = self.write_ptr + B
        if end > self.total_capacity:
            raise RuntimeError(f"Capacity exceeded: write_ptr={self.write_ptr} + B={B} > {self.total_capacity}")

        # write contiguous block
        self.mm[self.write_ptr:end, :] = emb.numpy().astype(self.dtype, copy=False)
        self.write_ptr = end

        # append IDs
        df_ids = pd.DataFrame({"dicom_id": list(map(str, dicom_ids)),
                               "ref_dicom_id": list(map(str, ref_ids))})
        # Try Parquet; fallback to CSV if pyarrow missing
        try:
            # append to parquet by chunk (no native append) -> write shard files and glob later OR use fastparquet
            # Simpler: keep CSV append by default (robust). If pyarrow present, switch below.
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq
            # We'll do simple shards: ids_000001.parquet, later concat when needed
            shard_idx = (self.write_ptr - len(df_ids))
            shard_name = self.root / f"ids_{shard_idx:09d}_{self.write_ptr-1:09d}.parquet"
            table = pa.Table.from_pandas(df_ids)
            pq.write_table(table, shard_name)
            # keep a tiny index file for shards (optional)
            with open(self.root / "ids_shards.txt", "a") as f:
                f.write(str(shard_name.name) + "\n")
        except Exception:
            # CSV append fallback
            header = not self.ids_csv.exists()
            df_ids.to_csv(self.ids_csv, index=False, mode="a", header=header)

    def flush(self):
        self.mm.flush()
        self._save_ckpt()

    def filled(self) -> int:
        return self.write_ptr

    def close(self):
        self.flush()


def main():
    # time stampfor output file
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default='/data/geraugi/biovil-t/test_biovil_t_dataset.csv')
    ap.add_argument("--out", default=f'/data/geraugi/biovil-t/inference/biovilt_results_{ts}.csv')
    ap.add_argument("--bad_dir", default=f'/data/geraugi/biovil-t/inference/bad_dir_{ts}')
#    ap.add_argument("--emb-dir", default=f'/data/geraugi/biovil-t/inference/embeddings_2025-08-11_07:29:04',
    ap.add_argument("--emb-dir", default=f'/data/geraugi/biovil-t/inference/embeddings_{ts}',
                    help="Directory to store memmapped embeddings + ids.")
    ap.add_argument("--emb-dtype", default="float32", choices=["float16", "float32"],
                    help="Disk dtype for embeddings.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume if checkpoint + memmap exist in --emb-dir.")
    ap.add_argument("--flush-every", type=int, default=500,
                    help="Flush embeddings + checkpoint every N samples.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--tau-add", type=float, default=0.05)
    ap.add_argument("--tau-miss", type=float, default=0.05)
    ap.add_argument("--delta-tie", type=float, default=0.0)
    ap.add_argument("-v", type=int, default=1, help="verbosity (0=warn,1=info)")
    ap.add_argument("--log-interval", type=int, default=164,
                help="Log progress every N batches (SLURM-friendly).")
    args = ap.parse_args()
    infer(args)

if __name__ == "__main__":
    main()
