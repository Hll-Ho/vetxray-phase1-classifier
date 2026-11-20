# VetXRay phase-1 classifier — frozen artifact package

Frozen artifact package for a seed-controlled, pre-registered evaluation of multi-label
thoracic-radiograph classification in dogs and cats: four loss arms and two backbone arms,
15 training runs, 3 seeds (42/43/44), evaluated on a held-out reporting split of 1,623
studies. The analysis plan and its four conservative gates were frozen **before** any
reporting-split readout; the frozen document is archived at
[10.5281/zenodo.24201580](https://doi.org/10.5281/zenodo.24201580).

Releases: `v1.0` — frozen artifact package (15 runs, 12 evaluable classes).

## 1. What is in here

| Path | Contents | Where it appears in the paper |
|---|---|---|
| `data/splits/train.csv` | 5,489 training studies (patient-disjoint from the other two lists) | §3 split definitions, Table 3.1 |
| `data/splits/calibration.csv` | 583 calibration studies | Table 4.1, Table 4.3 |
| `data/splits/reporting.csv` | 1,623 reporting studies (with GT label vectors) | Table 4.2, Table 4.4, Figure 1 |
| `outputs/run_logs/<tag>/` | per-run `log.txt`, `metrics.json`, `args.json` — 15 runs | Tables 4.1–4.3 |
| `outputs/eval/s4*_focal/per_class_report.csv` | per-class AP/F1 on the reporting split, 3 seeds | Table 4.4, Figure 1 |
| `outputs/eval/delta_matrix_583_arm_vs_focal.csv` | per-seed × per-class deltas vs the focal arm | §5 delta analysis |
| `figures/fig1_data.csv`, `figures/fig1_script.py` | source data and plotting script for Figure 1 | Figure 1 |
| `code/train_phase1.py` | training entry point (the exact 15-run version) | §3.2 |
| `code/w8_calibrate_eval.py` | calibration + reporting-split evaluation | §3.5 |
| `code/eval.py` | per-class AP/F1 and threshold sweep | §3.5, Table 4.4 |
| `code/RADDINO_LORA_RECIPE.json` | Rad-DINO + LoRA adaptation recipe (no weights) | §3.3 |
| `verify_split.py` | one-command integrity check of the three split lists | — |
| `PATH_NORMALIZATION.json` | per-file counts of local paths replaced by placeholders | — |

Run tags: `s{42,43,44}_{bce,focal,asl,pw}` (ConvNeXt-S arm, four loss configurations) and
`s{42,43,44}_raddino` (Rad-DINO + LoRA). Images are **not** redistributed; only identifiers
and split membership are.

## 2. Reproduction

```bash
# 0) environment: Python 3.10; CUDA 12.1 wheels
pip install -r requirements.txt            # torch==2.5.1+cu121 / torchvision==0.20.1+cu121
#    if the +cu121 builds are not resolvable from PyPI:
#    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
#        --index-url https://download.pytorch.org/whl/cu121

# 1) data: download the radiographs from 10.5281/zenodo.19051776,
#    then lay them out as <data-root>/images/<filename>
#    (the filename column of every split list matches the dataset filenames)

# 2) integrity of the released splits
python verify_split.py
python verify_split.py --manifest full_cases_7695.csv   # optional, stronger check

# 3) train one run (example: focal arm, seed 42)
python code/train_phase1.py --model convnext_small --loss focal \
    --labels  <data-root>/train_labels_subset.csv \
    --train-file <data-root>/train_minus_cal.csv \
    --val-file   <data-root>/cal.csv \
    --data-dir <data-root> --images-dir <data-root>/images \
    --out-root outputs/run_logs --tag s42_focal --size 512 \
    --batch-size 8 --grad-accum 2 --epochs 20 --patience 5 --seed 42

# 4) calibrate + evaluate on the reporting split
python code/w8_calibrate_eval.py --help

# 5) Figure 1
python figures/fig1_script.py     # prints Spearman rho and rewrites fig1.pdf/png
```

Notes on exact reproduction:

- **Path placeholders.** Absolute paths that occurred in run arguments and logs were
  replaced by `<VETPETS_ROOT>` / `<USER_HOME>` before release, and the corresponding
  defaults in `code/` were rewritten the same way. **Pass every path explicitly** as in
  step 3 above. No numeric value, timestamp, metric or log line was otherwise altered;
  counts are in `PATH_NORMALIZATION.json`.
- **Frozen readout.** The reported quantity is `epochs[-1].val_map`, not the best epoch
  (the stopping rule `min(20, best_epoch + 5)` never selects a maximum); `best_val_map`
  is retained in every `metrics.json` for sanity checks only.
- **Seeds.** The three seeds are 42/43/44; reported numbers are three-seed means, never
  best-of.
- **Rad-DINO.** Weights are research-only and are not redistributed; only the recipe is
  released (`code/RADDINO_LORA_RECIPE.json`), so the arm can be rebuilt from the upstream
  weights.
- `requirements.txt` is a literal `pip freeze` of the training environment; it pins
  transitive versions and may include packages not needed for inference.

## 3. Verification trail

- Pre-registration document (DOI `10.5281/zenodo.24201580`), SHA-256 of the canonical
  markdown `0cab8596fbf4fd561c7b1de3c975f0708f403cccfdaca652803dce926db21843`.
- The pre-registration record was downloaded back from Zenodo after publication and all
  three files re-hashed byte-for-byte.
- Split integrity: `verify_split.py` checks row counts (5,489 / 583 / 1,623, total 7,695),
  case-level and patient-level disjointness, and — given a dataset manifest — set equality
  with the manifest.
- Split derivation: from the dataset manifest (`case_id`, `split`), the reporting list is
  every `split = val` study (1,623); the calibration list is the fixed 583-study subset used
  during training; the training list is every remaining `split = train` study (5,489). All
  three are patient-disjoint.
- Figure 1 source data reproduces the paper's statistic on the 12 evaluable classes:
  Spearman rho = 0.615, p = 0.033.

## 4. Citation & license

If you use this package, please cite the pre-registration document and the dataset:

```bibtex
@misc{he2026vetxray_pre,
  author       = {He, Lian},
  title        = {Phase 1 Classifier Diagnosis and Plan v2.2 (pre-registration document)},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24201580},
  url          = {https://doi.org/10.5281/zenodo.24201580}
}

@misc{he2026vetxray_reports,
  author       = {He, Lian},
  title        = {W7-VetXRay-Report: 7,695 label-conditioned, model-generated
                  veterinary thoracic-radiograph reports},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24200927},
  url          = {https://doi.org/10.5281/zenodo.24200927}
}
```

Source data: VetXRay, DOI [10.5281/zenodo.19051776](https://doi.org/10.5281/zenodo.19051776)
(CC BY 4.0). Article link: `[arXiv link to be added after endorsement]` — the only
placeholder in this repository.

- **Code** (`code/`, `verify_split.py`): MIT, see `LICENSE`.
- **Derived data** (`data/`, `outputs/`, `figures/*.csv`): CC BY 4.0.
- **Model weights**: not distributed; see `NOTICE` for the Rad-DINO research-only terms.
