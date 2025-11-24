from typing import List, Dict, Any, Optional
import io, os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import math, sys, logging
from pathlib import Path
from torch.utils.data import get_worker_info

from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

class PairCXRDatasetLite(Dataset):
    """
    df columns (as you showed):
      dicom_id, ref_dicom_id,
      tar_path, member_path (opt), offset, size,
      ref_tar_path, ref_member_path (opt), ref_offset, ref_size,
      additional_*, missing_* (0/1)
    """
    def __init__(self, df_path='/data/geraugi/biovil-t/test_biovil_t_dataset.csv', diseases: Optional[List[str]] = None,
                 transform=None, bad_dir=None, skip_errors=True, logger=None):
        self.df = pd.read_csv(df_path)
        # get only 10 rows for testing
        self.df = self.df.iloc[:16389] 
        #self.df = self.df.head(10)
        #self.df = df.copy()
        if self.df.index.name != "dicom_id":
            self.df = self.df.set_index("dicom_id")
        self.df.index = self.df.index.astype(str)
        self.df["ref_dicom_id"] = self.df["ref_dicom_id"].astype(str)

        # infer diseases once; don't put names into samples
        if diseases is None:
            add = [c[len("additional_"):] for c in self.df.columns if c.startswith("additional_")]
            miss = [c[len("missing_"):]    for c in self.df.columns if c.startswith("missing_")]
            diseases = sorted(set(add) | set(miss))
        self.diseases = diseases
        self.add_cols  = [f"additional_{d}" for d in self.diseases]
        self.miss_cols = [f"missing_{d}"   for d in self.diseases]

        # required columns
        for c in ["tar_path","offset","size","ref_tar_path","ref_offset","ref_size"]:
            if c not in self.df.columns:
                raise ValueError(f"Missing column: {c}")

        self.transform = transform or create_chest_xray_transform_for_inference(512, 448)
        self.bad_dir = bad_dir
        self.skip_errors = skip_errors
        self.log = logger or logging.getLogger(__name__)
        if self.bad_dir:
            Path(self.bad_dir).mkdir(parents=True, exist_ok=True)

    def _log_bad(self, idx, row, exc):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        msg = (f"[bad-sample] wid={wid} idx={idx} err={type(exc).__name__}: {exc} | "
               f"ref=({row.get('ref_tar_path')},{row.get('ref_offset')},{row.get('ref_size')}) "
               f"cur=({row.get('cur_tar_path')},{row.get('cur_offset')},{row.get('cur_size')})")
        try:
            self.log.warning(msg)
        except Exception:
            print(msg, file=sys.stderr)

        if self.bad_dir:
            tsv = Path(self.bad_dir) / f"bad_samples_worker{wid}.tsv"
            with open(tsv, "a", encoding="utf-8") as f:
                f.write("\t".join(map(str, [
                    idx, type(exc).__name__, str(exc),
                    row.get("ref_tar_path"), row.get("ref_offset"), row.get("ref_size"),
                    row.get("cur_tar_path"), row.get("cur_offset"), row.get("cur_size"),
                ])) + "\n")

    @staticmethod
    def _read_from_tar(path: str, offset: int, size: int) -> Image.Image:
        if not isinstance(path, str) or not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            f.seek(int(offset)); raw = f.read(int(size))
        img = Image.open(io.BytesIO(raw)).convert("L")  # grayscale for HI-ML transform
        img.load()
        return img

    def __len__(self): return len(self.ids)

    def _is_missing(self, x):
            return (
                x is None
                or (isinstance(x, float) and math.isnan(x))
                or (isinstance(x, str) and (x.strip() == "" or x.strip().lower() == "nan"))
            )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        did = self.ids[idx]
        row = self.df.iloc[idx]
        try:
            # quick pre-check to avoid obvious failures
            keys = ("ref_tar_path","ref_offset","ref_size","tar_path","offset","size")
            if any(self._is_missing(row[k]) for k in keys):
                raise FileNotFoundError("Missing tar fields")

            prev = self._read_from_tar(row["ref_tar_path"], row["ref_offset"], row["ref_size"])
            curr = self._read_from_tar(row["tar_path"], row["offset"], row["size"])

            x_curr = self.transform(curr)   # [3,H,W]
            x_prev = self.transform(prev)   # [3,H,W]

            y_add  = torch.tensor(row[self.add_cols].astype(int).values,  dtype=torch.int8)
            y_miss = torch.tensor(row[self.miss_cols].astype(int).values, dtype=torch.int8)

            sample = {
                "dicom_id": did,
                "ref_dicom_id": row["ref_dicom_id"],
                "pixel_values_curr": x_curr,
                "pixel_values_prev": x_prev,
                "y_add": y_add,
                "y_miss": y_miss,
            }
            return sample

        except Exception as e:
            if not self.skip_errors:
                raise
            self._log_bad(idx, row, e)
            return None


#df_path = '/data/geraugi/biovil-t/test_biovil_t_dataset.csv'
#
## Assuming DataFrame 'df' is your testset with proper columns,
## and PairCXRDatasetLite is already imported or in the same file:
#
## Setup dataset and loader
#dataset = PairCXRDatasetLite(df_path)  # no ground truth
#dataset.ids = list(dataset.df.index.astype(str))  # ensure ids are set
#loader = DataLoader(dataset, batch_size=5, shuffle=False)
#
## Grab and print first batch
#batch = next(iter(loader))
#print("First batch keys:", batch.keys())
#print("dicom_id:", batch["dicom_id"])
#print("pixel_values_curr shape:", batch["pixel_values_curr"].shape)
#print("pixel_values_prev shape:", batch["pixel_values_prev"].shape)
#print("y_add:", batch["y_add"])
#print("y_miss:", batch["y_miss"])
#print("additional columns:", dataset.add_cols)
#print("missing columns:", dataset.miss_cols)
