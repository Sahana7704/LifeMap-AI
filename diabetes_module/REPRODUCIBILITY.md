# LifeMap AI Diabetes Module - Reproducibility Guide

This document provides exact specifications and deterministic instructions to reproduce all empirical findings, models, tables, figures, and reports for the LifeMap AI Diabetes Module.

---

## 1. Environment & Dependencies
- **Python Version:** Python 3.11.9 (64-bit)
- **Virtual Environment Path:** `diabetes_module/.venv-diabetes`
- **Dependency Freeze:** Saved in [requirements_frozen.txt](file:///diabetes_module/reports/paper/requirements_frozen.txt)
- **Random Seed:** `SEED = 42` is fixed globally across NumPy, PyTorch, Scikit-Learn, and XGBoost.

---

## 2. Raw Data Sources & CDC URLs
All data originate from official CDC public health portals:

### NHANES (2013-2018 Cycles H, I, J)
- **Demographics (DEMO):**
  - Cycle H (2013-14): https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT
  - Cycle I (2015-16): https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT
  - Cycle J (2017-18): https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT
- **Body Measures (BMX):**
  - Cycle H: https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BMX_H.XPT
  - Cycle I: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.XPT
  - Cycle J: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT
- **Blood Pressure Examination (BPX):**
  - Cycle H: https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BPX_H.XPT
  - Cycle I: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.XPT
  - Cycle J: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT
- **Blood Pressure & Cholesterol Questionnaire (BPQ):**
  - Cycle H: https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BPQ_H.XPT
  - Cycle I: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPQ_I.XPT
  - Cycle J: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPQ_J.XPT
- **Diabetes Questionnaire (DIQ):**
  - Cycle H: https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DIQ_H.XPT
  - Cycle I: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DIQ_I.XPT
  - Cycle J: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.XPT
- **Smoking Cigarette Use (SMQ):**
  - Cycle H: https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/SMQ_H.XPT
  - Cycle I: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/SMQ_I.XPT
  - Cycle J: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/SMQ_J.XPT
- **Physical Activity (PAQ):**
  - Cycle H: https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/PAQ_H.XPT
  - Cycle I: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/PAQ_I.XPT
  - Cycle J: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT
- **Medical Conditions - Family History (MCQ):**
  - Cycle H: https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/MCQ_H.XPT
  - Cycle I: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/MCQ_I.XPT
  - Cycle J: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/MCQ_J.XPT
- **Glycated Hemoglobin Lab (GHB):**
  - Cycle H: https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/GHB_H.XPT
  - Cycle I: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/GHB_I.XPT
  - Cycle J: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.XPT

### CDC BRFSS (2015 Annual Survey)
- CDC Survey Data & Documentation: https://www.cdc.gov/brfss/annual_data/annual_2015.html
- Curated Behavioral Indicators: Hosted via Kaggle CDC BRFSS 2015 dataset (`archive (2)/diabetes_012_health_indicators_BRFSS2015.csv`).

---

## 3. SHA-256 Cryptographic Checksums of Raw Files

| File Name | SHA-256 Checksum |
| :--- | :--- |
| `BMX_H.xpt` | `fd5e9fc6e6aab0a4aee6e699f51497bbc9b62101f7f43aee924c473e38fd9442` |
| `BMX_I.xpt` | `d31da84e14212b4e58e8340598a5b8e2144fac83333563e966eb1b332e4141d6` |
| `BMX_J.xpt` | `8d675e42d8826ac98714b2c3dd4c5138a5e353fb4424f7eff5e6db4a01ce838a` |
| `BPQ_H.xpt` | `aa6ddeab80c73b074ccfd04ad010da404ecb91c048476a194594ce9552d03d49` |
| `BPQ_I.xpt` | `7f7e51cd0497eb648f64eaafaf7ed07825d617c32b2cd6853ed249fce1143a38` |
| `BPQ_J.xpt` | `63cfe1c331a1e7d3534328ac86312a21ffee3e01aa189bc1f74d06855239e5aa` |
| `BPX_H.xpt` | `b6641efc30ac9b9de328dfbf4110d0d120439aadca2c2bc89f2e77310d1a0c8e` |
| `BPX_I.xpt` | `45d5c9fa101ccc6da82241bd45c99611595e699665eed2cfed232e9070b52cc1` |
| `BPX_J.xpt` | `f98f1d9a4c18173e715c749b4294f6f1525140d5ad5751316bb17cc038fb3423` |
| `DEMO_H.xpt` | `f8f0cbb3085a323d4cde22349b164878fea1e64dbc404e65b5815c7816b547d7` |
| `DEMO_I.xpt` | `c9297c6c37ae8f78f29be9568fa2a03cf3b112616a39afee04030fc775a66a0d` |
| `DEMO_J.xpt` | `c0b46e0345ea19404928656277c8b0d10b0cca348a9b2fe4fc3c67e8b7ee73ec` |
| `DIQ_H.xpt` | `c74c7ccef65e6997dfac1db1e73bc3f63dec67c70b2e322ffc14f14ce27429a9` |
| `DIQ_I.xpt` | `e87587479b29f175b63eee5dd40d582837e3e3fe2665503012085eefdb978e0d` |
| `DIQ_J.xpt` | `1ecbf5360dfc331d1efbf32198553dc30e9a1f4cff0a907ed30ca72bac797f89` |
| `GHB_H.xpt` | `0695894ad55ac96f315a8415401977b0856c402d16762115b534bcd5dfeae89e` |
| `GHB_I.xpt` | `e4bc626cd12f6057c7806aef4f85874c9bf1407a7480d6621a4af142260addd2` |
| `GHB_J.xpt` | `35f07094573a0061a03ed609a5a363b34eb1b1c7065d1623b43d72e132a8a654` |
| `MCQ_H.xpt` | `5c83210e6e1fe929282621322ed0f94d95cdb0a37ccaff36f703ab0b2e7b9dc6` |
| `MCQ_I.xpt` | `8b2bd62f5bcd5820d6aac380e4bad233d22ecab42fcf4a3a9a023ffc6b9f4002` |
| `MCQ_J.xpt` | `79c50c805a377fcd61d44fae8b86a16af0a5741f8b1aeebda918d7d743c68f98` |
| `PAQ_H.xpt` | `fa35acbb489594e8e55350f7402a12b5a607805c28ec16246929d33237bd3269` |
| `PAQ_I.xpt` | `c508dc563a907b2fcd05048f37008197257a3e1c6eb64563623c60459c1afc9d` |
| `PAQ_J.xpt` | `048120aa3a5c8b5cd457ad3e13dfd62d728b12ea6644960e933ad374ec907e79` |
| `SMQ_H.xpt` | `86d74bb8842ce6d60350944417cbd3e9ee39dd5a9fa1bda33426f276b673861f` |
| `SMQ_I.xpt` | `529ce1c407e99dce937ca75d14c8c630e0c9c30a4a9a851760cd2884bbb28fe0` |
| `SMQ_J.xpt` | `73c3708e540cbf520a566508aa629c3e4ba53cad77dae9388ca36384646455f4` |
| `BRFSS2015.csv` | `a971809d0786d2d7d7f6070f83dfe8af9d860ea33a79e0e2701c8a49f78f9861` |

---

## 4. Execution Order & Pipeline Workflow

To reproduce all models, predictions, paper tables, and visual assets from scratch:

```bash
# Step 1: Preprocess raw NHANES and BRFSS data into leakage-free datasets
python diabetes_module/src/preprocess.py

# Step 2: Download and harmonize glycated hemoglobin (HbA1c) laboratory files
python diabetes_module/src/download_and_verify_ghb.py

# Step 3: Train Tier 1 models, tune hyperparameters, perform stacking & calibration
python diabetes_module/src/train_and_evaluate.py

# Step 4: Train Tier 2 models, perform ablation studies, and save Tier 2 predictions
python diabetes_module/src/tier2_train_and_evaluate.py

# Step 5: Execute robustness checks (5-fold CV, temporal split, complete-case analysis)
python diabetes_module/src/robustness_checks.py

# Step 6: Generate all paper tables (CSV/LaTeX) and figures (PNG/PDF)
python scratch/generate_all_artifacts.py

# Step 7: Execute the final report notebook and export self-contained HTML
python scratch/build_and_run_notebook.py
```

---

## 5. Verification Commands

Verify the pipeline integrity, disjoint splits, and lack of leakage:
```bash
python -c "
import json
with open('diabetes_module/models/split_indices.json') as f:
    s = json.load(f)
tr, va, te = set(s['train_indices']), set(s['val_indices']), set(s['test_indices'])
assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0
print('Split Integrity Verified: Zero Overlap.')
"
```
