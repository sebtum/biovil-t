import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple, Union, List

Label = np.ndarray  # shape [N], values in {+1 (additional), -1 (missing), 0 (no-output)}

def decide_from_margins(
    s_add: np.ndarray, s_miss: np.ndarray,
    tau_add: float, tau_miss: float, delta_tie: float = 0.0
) -> Label:
    """
    Vectorized decision:
      - predict +1 ("additional") if s_add >= tau_add and s_miss < tau_miss
      - predict -1 ("missing")   if s_miss >= tau_miss and s_add  < tau_add
      - abstain (0) if both below thresholds
      - if both >= thresholds: choose larger margin, unless |gap|<delta_tie -> abstain
    """
    s_add = np.asarray(s_add); s_miss = np.asarray(s_miss)
    add_ok  = s_add >= tau_add
    miss_ok = s_miss >= tau_miss

    pred = np.zeros_like(s_add, dtype=np.int8)
    # exclusive positives
    pred[(add_ok)  & (~miss_ok)] = +1
    pred[(miss_ok) & (~add_ok)]  = -1

    # both true -> tie break
    both = add_ok & miss_ok
    if both.any():
        gap = s_add[both] - s_miss[both]
        choose_add  = gap >  +delta_tie
        choose_miss = gap <  -delta_tie
        idx = np.where(both)[0]
        pred[idx[choose_add]]  = +1
        pred[idx[choose_miss]] = -1
        # if |gap|<=delta_tie remain 0 (abstain)

    # --- Count cases ---
    exclusive_add    = np.sum((add_ok) & (~miss_ok))
    exclusive_miss   = np.sum((miss_ok) & (~add_ok))
    tie_cases        = np.sum(both & (np.abs(s_add - s_miss) <= delta_tie))
    abstain_cases    = np.sum(pred == 0)

    #print(f"Exclusive additional: {exclusive_add}")
    #print(f"Exclusive missing:    {exclusive_miss}")
    #print(f"Tie cases:            {tie_cases}")
    #print(f"Abstain total:        {abstain_cases}")

    return pred

def _safe_div(num: int, den: int) -> float:
    return (num / den) if den > 0 else 0.0

def pr_for_class(pred: Label, y_pos: np.ndarray, pos_code: int) -> Tuple[float, float, Dict[str,int]]:
    """
    Metrics for a single class:
      - pos_code = +1 for 'additional', -1 for 'missing'
      - y_pos: 0/1 ground-truth vector for the same class
    """
    y_pos = np.asarray(y_pos).astype(bool)
    pred_pos = (pred == pos_code)

    tp = int(np.sum(pred_pos & y_pos))
    fp = int(np.sum(pred_pos & (~y_pos)))
    fn = int(np.sum((~pred_pos) & y_pos))

    prec = _safe_div(tp, tp + fp)
    rec  = _safe_div(tp, tp + fn)
    return prec, rec, {"tp": tp, "fp": fp, "fn": fn, "support": int(np.sum(y_pos))}

def evaluate_at_thresholds(
    df: pd.DataFrame,
    diseases: List[str],
    tau_add: float,
    tau_miss: float,
    delta_tie: float = 0.0,
) -> pd.DataFrame:
    """
    Expects columns in df:
      margin_add_{d}, margin_miss_{d}, gt_additional_{d}, gt_missing_{d}
    Returns a tidy DataFrame with precision/recall per disease & class.
    """
    rows = []
    for d in diseases:
        s_add  = df[f"margin_add_{d}"].to_numpy()
        s_miss = df[f"margin_miss_{d}"].to_numpy()
        y_add  = df.get(f"gt_additional_{d}")
        y_miss = df.get(f"gt_missing_{d}")
        if y_add is None or y_miss is None:
            raise ValueError(f"Ground truth columns missing for disease '{d}'.")

        pred = decide_from_margins(s_add, s_miss, tau_add, tau_miss, delta_tie)

        p_add, r_add, c_add   = pr_for_class(pred, y_add.to_numpy(),  +1)
        p_miss, r_miss, c_miss = pr_for_class(pred, y_miss.to_numpy(), -1)

        rows.append({
            "disease": d, "class": "additional",
            "precision": p_add, "recall": r_add, **c_add,
            "tau_add": tau_add, "tau_miss": tau_miss, "delta_tie": delta_tie
        })
        rows.append({
            "disease": d, "class": "missing",
            "precision": p_miss, "recall": r_miss, **c_miss,
            "tau_add": tau_add, "tau_miss": tau_miss, "delta_tie": delta_tie
        })
    return pd.DataFrame(rows)

def sweep_thresholds(
    df: pd.DataFrame,
    diseases: List[str],
    tau_add_grid: Iterable[float],
    tau_miss_grid: Iterable[float],
    delta_tie_grid: Iterable[float] = (0.0,),   # <— allow sweeping delta_tie too
) -> pd.DataFrame:
    all_rows = []
    for dt in delta_tie_grid:
        for ta in tau_add_grid:
            for tm in tau_miss_grid:
                res = evaluate_at_thresholds(df, diseases, ta, tm, delta_tie=dt)
                all_rows.append(res)
    out = pd.concat(all_rows, ignore_index=True)
    return out

def fbeta_series(prec: pd.Series, rec: pd.Series, beta: float = 0.5) -> pd.Series:
    beta2 = beta * beta
    denom = beta2 * prec + rec
    num   = (1.0 + beta2) * (prec * rec)
    out = pd.Series(np.zeros(len(prec), dtype=float), index=prec.index)
    mask = denom > 0
    out.loc[mask] = num.loc[mask] / denom.loc[mask]
    return out

# Suppose 'preds.csv' is what inference.py saved (with margin_* and gt_* columns)
ts = "2025-08-11_17:46:22" 
ts = "2025-08-12_00:47:28"
results_file = f"/data/geraugi/biovil-t/inference/biovilt_results_{ts}.csv"
results = pd.read_csv(results_file)
print(f"Loaded results from with {len(results)} rows.")

# Use the same disease list your dataset had:
diseases = [c.replace("gt_additional_", "") for c in results.columns if c.startswith("gt_additional_")]

# 1) Single operating point
res = evaluate_at_thresholds(results, diseases, tau_add=0.10, tau_miss=0.08, delta_tie=0.01)
print(res.sort_values(["disease","class"])[["disease","class","precision","recall","support"]])

## 2) Sweep to pick thresholds (e.g., target high precision)
tau_add_grid  = np.linspace(0.00, 0.50, 11)   # 0.01 steps
tau_miss_grid = np.linspace(0.00, 0.50, 11)
delta_tie_grid = (0.00, 0.01, 0.02, 0.05)

print(f"Sweeping thresholds")
grid = sweep_thresholds(
    results, diseases,
    tau_add_grid=tau_add_grid,
    tau_miss_grid=tau_miss_grid,
    delta_tie_grid=delta_tie_grid,   # requires the expanded version I sent
)
grid["f05"] = fbeta_series(grid["precision"], grid["recall"], beta=0.5)

# Pick the best (highest F0.5) per (disease, class).
# Tie-break: prefer higher precision, then higher recall.
best_f05 = (
    grid.sort_values(
        ["disease", "class", "f05", "precision", "recall"],
        ascending=[True, True, False, False, False]
    )
    .drop_duplicates(subset=["disease", "class"], keep="first")
    .sort_values(["disease", "class"])
    .reset_index(drop=True)
)

# Keep the useful columns
best_f05 = best_f05[[
    "disease", "class", "f05", "precision", "recall",
    "tp", "fp", "fn", "support",
    "tau_add", "tau_miss", "delta_tie"
]]

print(best_f05)
# Optionally save:

best_f05_file = f"/data/geraugi/biovil-t/inference/best_f05_{ts}.csv"
print(f"Best F0.5 results saved to {best_f05_file}")
best_f05.to_csv(best_f05_file, index=False)
#res.to_csv(f"/data/geraugi/biovil-t/inference/eval_single_point_{ts}.csv", index=False)
#grid.to_csv(f"/data/geraugi/biovil-t/inference/eval_sweep_{ts}.csv", index=False)
