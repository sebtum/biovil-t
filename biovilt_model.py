# biovilt_model.py
from __future__ import annotations
from typing import Dict, List, Tuple
import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder

log = logging.getLogger(__name__)

# ---------- Image (pair) encoder with caching ----------

class BioviltPairEncoder:
    def __init__(self, device: str = None, half: bool = False, enable_cache: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = get_biovil_t_image_encoder().to(self.device).eval()
        self.half = bool(half) and self.device.type == "cuda"
        if self.half:
            self.model.half()
        # simple in-memory cache for pair embeddings
        self.enable_cache = enable_cache
        self._pair_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self):
        self._pair_cache.clear()
        self.cache_hits = self.cache_misses = 0

    @torch.no_grad()
    def encode_pair(
        self,
        x_curr: torch.Tensor, x_prev: torch.Tensor,
        dicom_id: List[str] | None = None, ref_dicom_id: List[str] | None = None,
    ) -> torch.Tensor:
        """
        x_curr/x_prev: [B,3,H,W] (CPU or GPU). If IDs provided, caching is enabled per (dicom, ref_dicom).
        Returns: [B, D] L2-normalized embeddings on self.device.
        """
        B = x_curr.shape[0]
        out_list: List[torch.Tensor] = []

        # Try to reuse cached pairs
        idxs_to_compute = list(range(B))
        if self.enable_cache and dicom_id is not None and ref_dicom_id is not None:
            for i in range(B):
                key = (str(dicom_id[i]), str(ref_dicom_id[i]))
                if key in self._pair_cache:
                    self.cache_hits += 1
                    out_list.append(self._pair_cache[key].to(self.device, non_blocking=True))
                else:
                    self.cache_misses += 1
                    out_list.append(None)

            idxs_to_compute = [i for i, e in enumerate(out_list) if e is None]

        # Compute missing ones
        if idxs_to_compute:
            xc = x_curr[idxs_to_compute].to(self.device, non_blocking=True)
            xp = x_prev[idxs_to_compute].to(self.device, non_blocking=True)
            if self.half:
                xc = xc.half(); xp = xp.half()

            try:
                o = self.model.forward(current_image=xc, previous_image=xp)
            except TypeError:
                patch_x, pooled_x = self.model.encoder(
                    current_image=xc, previous_image=xp, return_patch_embeddings=True
                )
                o = self.model.forward_post_encoder(patch_x, pooled_x)

            e = F.normalize(o.projected_global_embedding, dim=-1)  # [N, D]

            # Fill slots and cache
            j = 0
            for i in idxs_to_compute:
                out_list[i] = e[j]
                if self.enable_cache and dicom_id is not None and ref_dicom_id is not None:
                    key = (str(dicom_id[i]), str(ref_dicom_id[i]))
                    # store on CPU to limit VRAM footprint
                    self._pair_cache[key] = e[j].detach().cpu()
                j += 1

        # Stack in the original order
        return torch.stack(out_list, dim=0)  # [B, D]

# ---------- Text encoder with prompt cache ----------

class BioviltText:
    """
    Loads BioViL-T text side from HF using safetensors (avoids torch.load gate).
    Caches prompt embeddings so phrases are encoded once.
    """
    def __init__(self, device: str = None, model_id: str = "microsoft/BiomedVLP-BioViL-T"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, use_safetensors=True
        ).to(self.device).eval()
        self._cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    @torch.no_grad()
    def embed_prompts(self, prompts: List[str]) -> torch.Tensor:
        key = tuple(prompts)
        if key in self._cache:
            return self._cache[key].to(self.device, non_blocking=True)
        toks = self.tok(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        te = self.model.get_projected_text_embeddings(
            input_ids=toks.input_ids, attention_mask=toks.attention_mask
        )
        te = F.normalize(te, dim=-1)  # [K, D]
        # cache on CPU
        self._cache[key] = te.detach().cpu()
        return te

# ---------- Prompts, scoring, decisions ----------

def build_prompt_bank(text: BioviltText, disease: str) -> Dict[str, torch.Tensor]:
    add_pos  = [f"new {disease}", f"interval appearance of {disease}"]
    add_neg  = [f"no new {disease}", f"no interval appearance of {disease}"]
    miss_pos = [f"resolved {disease}", f"{disease} cleared", f"{disease} no longer present", f"interval resolution of {disease}"]
    miss_neg = [f"persistent {disease}", f"{disease} still present", f"{disease} present", f"no interval resolution of {disease}"]
#    add_pos  = [f"new {disease}", f"increased {disease}", f"additional {disease}"]
#    add_neg  = [f"no new {disease}", f"no interval increase of {disease}"]
#    miss_pos = [f"resolved {disease}", f"{disease} decreased", f"{disease} no longer present"]
#    miss_neg = [f"persistent {disease}", f"{disease} unchanged"]
    print(f"Positive prompts for additional '{disease}': {add_pos}")
    print(f"Positive prompts for missing '{disease}': {miss_pos}")
    print(f"Negative prompts for additional '{disease}': {add_neg}")
    print(f"Negative prompts for missing '{disease}': {miss_neg}")

    return {
        "add_pos":  text.embed_prompts(add_pos),
        "add_neg":  text.embed_prompts(add_neg),
        "miss_pos": text.embed_prompts(miss_pos),
        "miss_neg": text.embed_prompts(miss_neg),
    }

@torch.no_grad()
def score_margins_pairwise(e_pair: torch.Tensor, bank, agg: str = "mean", T: float = 0.1):
    """
    bank must contain aligned pairs with the same number of prompts per side.
    Example layout:
      bank["add_pos"]  : [K, D]  (K pos prompts)
      bank["add_neg"]  : [K, D]  (K matching neg prompts)
      bank["miss_pos"] : [K2, D]
      bank["miss_neg"] : [K2, D]
    """
    # [B, K]
    s_add_pos  = e_pair @ bank["add_pos"].T
    s_add_neg  = e_pair @ bank["add_neg"].T
    s_miss_pos = e_pair @ bank["miss_pos"].T
    s_miss_neg = e_pair @ bank["miss_neg"].T

    # per-pair margins [B, K] / [B, K2]
    m_add  = s_add_pos  - s_add_neg
    m_miss = s_miss_pos - s_miss_neg

    # aggregate per axis
    if agg == "mean":
        s_add  = m_add.mean(dim=1)
        s_miss = m_miss.mean(dim=1)
    elif agg == "lse":  # softmax-style aggregation
        s_add  = (torch.logsumexp(m_add / T,  dim=1)) * T
        s_miss = (torch.logsumexp(m_miss / T, dim=1)) * T
    elif agg == "max":
        s_add  = m_add.max(dim=1).values
        s_miss = m_miss.max(dim=1).values
    else:
        raise ValueError("agg must be one of {'mean','lse','max'}")
    return s_add, s_miss

def decide(s_add: torch.Tensor, s_miss: torch.Tensor,
           tau_add: float, tau_miss: float, delta_tie: float = 0.0) -> List[str | None]:
    add_ok  = s_add  >= tau_add
    miss_ok = s_miss >= tau_miss
    labs: List[str | None] = []
    for a_ok, m_ok, a, m in zip(add_ok.tolist(), miss_ok.tolist(), s_add.tolist(), s_miss.tolist()):
        if a_ok and not m_ok:   labs.append("additional")
        elif m_ok and not a_ok: labs.append("missing")
        elif (not a_ok) and (not m_ok): labs.append(None)
        else:
            labs.append(None if abs(a - m) < delta_tie else ("additional" if a > m else "missing"))
    return labs
