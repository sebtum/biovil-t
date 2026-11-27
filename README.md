# BioViL-T for new and resolved findings on MIMIC-Diff-VQA

Given a current and a prior chest X-ray, which diseases are **new** and which are **gone**? This repository tests [BioViL-T](https://arxiv.org/abs/2301.04558) as a zero-shot answer to that question. It is one of three context sources compared in my M.Sc. thesis, **[Context-Aware Difference VQA for Chest X-rays](https://github.com/sebtum/context-aware-diffvqa)**.

## What I built

| Part | File |
|---|---|
| Image-pair and text encoders with caching, prompt bank, scoring, decision rule | `biovilt_model.py` |
| Dataset that reads both images from tar archives | `pair_cxr_dataset_lite.py` |
| Resumable inference that stores pair embeddings and per-disease scores | `inference.py` |
| Threshold sweep, precision, recall and F0.5 | `evaluation.py` |
| Builds the pair dataset with additional and missing labels | `testing_biovil_t.ipynb` |

## Method

Each disease gets two scores, one for "additional" and one for "missing". A score is the mean difference between positive and negative prompt similarities.

| Class | Positive prompts | Negative prompts |
|---|---|---|
| Additional | new X; interval appearance of X | no new X; no interval appearance of X |
| Missing | resolved X; X cleared; X no longer present; interval resolution of X | persistent X; X still present; X present; no interval resolution of X |

The two classes are exclusive (thesis Table 3.1):

| Additional score | Missing score | Hint |
|---|---|---|
| < 0 | < 0 | none |
| < 0 | >= 0 | missing |
| >= 0 | < 0 | additional |
| >= 0 | >= 0, not above the additional score | additional |
| >= 0 | >= 0, above the additional score | missing |

The code uses adjustable thresholds in place of 0 (default 0.05, swept from 0 to 0.5). On an exact tie it does not pick "additional" as the table does.

## Result

Average F0.5 (an F-score that favours precision) after sweeping per-disease thresholds (thesis Table 5.11):

| Disease | Average F0.5 |
|---|---|
| Atelectasis | 0.2374 |
| Cardiomegaly | 0.1624 |
| Consolidation | 0.1051 |
| Edema | 0.2037 |
| Lung opacity | 0.2375 |
| Pleural effusion | 0.2354 |
| Pneumonia | 0.1369 |
| Pneumothorax | 0.0423 |

No disease reached the 0.5 that the thesis treats as a usable hint, so BioViL-T is not part of the final system. The thesis gives likely reasons: BioViL-T was trained on progression (stable, worsening, improving), which only partly matches new and resolved findings, and on single-disease examples. The prompt search was also short.

## Run it

This is research code. Paths are hard-coded to the thesis cluster, so edit them first. You need credentialed PhysioNet access to [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and [MIMIC-Diff-VQA](https://physionet.org/content/medical-diff-vqa/1.0.0/). These steps were read from the code and not re-run for this page.

```bash
conda env create -f env.yml
conda activate biovil-t-env
pip install hi-ml-multimodal pandas tqdm   # imported by the code, not listed in env.yml
```

1. Build `test_biovil_t_dataset.csv` with the dataset cells in `testing_biovil_t.ipynb`. They need the image index from the CheXFound adaptation, the PLURAL context file and a ground-truth difference file. No script here creates the last two.
2. Run inference:

   ```bash
   python inference.py --csv test_biovil_t_dataset.csv --out results.csv \
     --bad_dir bad_samples --emb-dir embeddings
   ```

3. In `evaluation.py`, point `results_file` and `best_f05_file` at your paths, then run `python evaluation.py`. It reports the best F0.5 per disease and class.

The dataset class reads at most 16,389 rows.

## Credits

| Work | Role here |
|---|---|
| [BioViL-T](https://arxiv.org/abs/2301.04558), Bannur et al. | Temporal vision-language model |
| [hi-ml-multimodal](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal) and [model weights](https://huggingface.co/microsoft/BiomedVLP-BioViL-T), Microsoft | Image and text encoders |
| [Xplainer](https://github.com/ChantalMP/Xplainer), Pellegrini et al. | Positive and negative prompt scoring idea |
