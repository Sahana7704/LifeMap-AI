"""
Data Preprocessing Pipeline for LifeMap AI Diabetes Module.
Processes raw NHANES (2013-2018 cycles) and BRFSS 2015 datasets into clean,
leakage-free datasets for model training and external validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def verify_nhanes_files(raw_dir: Path) -> dict[str, bool]:
    """Verify all 24 required NHANES .XPT files exist."""
    cycles = ["H", "I", "J"]
    modules = ["DEMO", "BMX", "BPX", "BPQ", "DIQ", "SMQ", "PAQ", "MCQ"]
    status = {}
    all_exist = True
    print("\n--- [Step 1] Verifying NHANES Raw Files ---")
    for m in modules:
        for c in cycles:
            fname = f"{m}_{c}.xpt"
            fpath = raw_dir / fname
            exists = fpath.exists()
            status[fname] = exists
            if not exists:
                print(f"  [MISSING] {fname}")
                all_exist = False
    if all_exist:
        print(f"  [SUCCESS] All {len(status)} NHANES .XPT files verified present in {raw_dir}")
    return status


def age_to_brfss_bucket(age: float) -> int:
    """Map age in years to BRFSS 13 age categories:
    1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44,
    6: 45-49, 7: 50-54, 8: 55-59, 9: 60-64, 10: 65-69,
    11: 70-74, 12: 75-79, 13: 80+
    """
    if pd.isna(age):
        return np.nan
    age = float(age)
    if age < 25:
        return 1
    elif age < 30:
        return 2
    elif age < 35:
        return 3
    elif age < 40:
        return 4
    elif age < 45:
        return 5
    elif age < 50:
        return 6
    elif age < 55:
        return 7
    elif age < 60:
        return 8
    elif age < 65:
        return 9
    elif age < 70:
        return 10
    elif age < 75:
        return 11
    elif age < 80:
        return 12
    else:
        return 13


def recode_binary_79_as_nan(val: float) -> float:
    """Recode NHANES 1 (Yes) -> 1, 2 (No) -> 0, 7 (Refused) / 9 (Don't know) -> NaN."""
    if pd.isna(val):
        return np.nan
    val = float(val)
    if val == 1.0:
        return 1.0
    elif val == 2.0:
        return 0.0
    else:
        return np.nan


def compute_phys_activity(df: pd.DataFrame, pa_cols: list[str]) -> pd.Series:
    """
    Physical Activity Recoding Rule:
    - 1.0 if ANY of PAQ605/620/635/650/665 == 1 (Yes).
    - 0.0 if NONE is 1, but at least ONE valid answer (2 = No) exists.
    - NaN if ALL are missing (NaN) or only invalid responses (7 = Refused, 9 = Don't know).
    """
    sub = df[pa_cols]
    has_1 = (sub == 1.0).any(axis=1)
    has_2 = (sub == 2.0).any(axis=1)

    result = pd.Series(np.nan, index=df.index, dtype=float)
    result[has_2 & ~has_1] = 0.0
    result[has_1] = 1.0
    return result


def process_nhanes(raw_dir: Path) -> pd.DataFrame:
    """Read, merge, harmonize, and clean NHANES adult cohort across cycles H, I, J."""
    cycles = ["H", "I", "J"]
    cycle_years = {"H": "2013-2014", "I": "2015-2016", "J": "2017-2018"}
    cycle_dfs = []

    print("\n--- [Step 2] Processing NHANES Data ---")

    for c in cycles:
        print(f"\nProcessing Cycle {c} ({cycle_years[c]}):")
        # 1. DEMO
        demo_path = raw_dir / f"DEMO_{c}.xpt"
        demo = pd.read_sas(demo_path, format="xport")
        print(f"  DEMO_{c}: shape={demo.shape}, columns={sorted(demo.columns.tolist())[:8]}...")
        # Adults RIDAGEYR >= 18
        adults = demo[demo["RIDAGEYR"] >= 18].copy()
        print(f"  Adults (RIDAGEYR >= 18): {len(adults)} out of {len(demo)}")
        merged = adults[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()

        # 2. BMX
        bmx = pd.read_sas(raw_dir / f"BMX_{c}.xpt", format="xport")
        print(f"  BMX_{c}: shape={bmx.shape}, columns={sorted(bmx.columns.tolist())[:6]}...")
        merged = merged.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")

        # 3. BPX
        bpx = pd.read_sas(raw_dir / f"BPX_{c}.xpt", format="xport")
        bpx_cols = ["SEQN"] + [f"BPXSY{i}" for i in range(1, 5) if f"BPXSY{i}" in bpx.columns] + \
                              [f"BPXDI{i}" for i in range(1, 5) if f"BPXDI{i}" in bpx.columns]
        print(f"  BPX_{c}: shape={bpx.shape}, selected BP cols={bpx_cols}")
        merged = merged.merge(bpx[bpx_cols], on="SEQN", how="left")

        # 4. BPQ
        bpq = pd.read_sas(raw_dir / f"BPQ_{c}.xpt", format="xport")
        print(f"  BPQ_{c}: shape={bpq.shape}, target cols=['BPQ020', 'BPQ080']")
        merged = merged.merge(bpq[["SEQN", "BPQ020", "BPQ080"]], on="SEQN", how="left")

        # 5. DIQ
        diq = pd.read_sas(raw_dir / f"DIQ_{c}.xpt", format="xport")
        print(f"  DIQ_{c}: shape={diq.shape}, target col=['DIQ010']")
        merged = merged.merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")

        # 6. SMQ
        smq = pd.read_sas(raw_dir / f"SMQ_{c}.xpt", format="xport")
        print(f"  SMQ_{c}: shape={smq.shape}, target col=['SMQ020']")
        merged = merged.merge(smq[["SEQN", "SMQ020"]], on="SEQN", how="left")

        # 7. PAQ
        paq = pd.read_sas(raw_dir / f"PAQ_{c}.xpt", format="xport")
        pa_target_cols = [col for col in ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"] if col in paq.columns]
        print(f"  PAQ_{c}: shape={paq.shape}, selected PA cols={pa_target_cols}")
        merged = merged.merge(paq[["SEQN"] + pa_target_cols], on="SEQN", how="left")

        # 8. MCQ
        mcq = pd.read_sas(raw_dir / f"MCQ_{c}.xpt", format="xport")
        mcq_target_cols = [col for col in ["MCQ300C"] if col in mcq.columns]
        print(f"  MCQ_{c}: shape={mcq.shape}, selected MCQ cols={mcq_target_cols}")
        merged = merged.merge(mcq[["SEQN"] + mcq_target_cols], on="SEQN", how="left")

        merged["cycle"] = c
        cycle_dfs.append(merged)

    # Concatenate all cycles
    nhanes_all = pd.concat(cycle_dfs, ignore_index=True)
    print(f"\nTotal merged adult respondents across 3 cycles: {nhanes_all.shape}")

    # Label extraction:
    # DIQ010 == 1 -> 1 (diabetes)
    # DIQ010 == 2 -> 0 (no diabetes)
    # Drop 3 (borderline), 7 (refused), 9 (don't know), and missing
    valid_label_mask = nhanes_all["DIQ010"].isin([1.0, 2.0])
    dropped_counts = nhanes_all["DIQ010"].value_counts(dropna=False)
    print("\nDIQ010 distribution before filtering:")
    print(dropped_counts)
    print(f"Retaining {valid_label_mask.sum()} respondents with clear diabetes status (1 or 2).")

    df_clean = nhanes_all[valid_label_mask].copy()
    df_clean["diabetes"] = (df_clean["DIQ010"] == 1.0).astype(int)

    # Feature Harmonization:
    # 1. Sex: RIAGENDR (1 male -> 1, 2 female -> 0)
    df_clean["sex"] = df_clean["RIAGENDR"].map({1.0: 1, 2.0: 0})

    # 2. Age: continuous RIDAGEYR
    df_clean["age"] = df_clean["RIDAGEYR"].astype(float)

    # 3. Age Bucket (13 BRFSS buckets)
    df_clean["age_bucket"] = df_clean["age"].apply(age_to_brfss_bucket)

    # 4. BMI: BMXBMI
    df_clean["bmi"] = df_clean["BMXBMI"].astype(float)

    # 5. Blood Pressure: mean systolic BP & mean diastolic BP
    sy_cols = [f"BPXSY{i}" for i in range(1, 5) if f"BPXSY{i}" in df_clean.columns]
    di_cols = [f"BPXDI{i}" for i in range(1, 5) if f"BPXDI{i}" in df_clean.columns]
    df_clean["mean_sbp"] = df_clean[sy_cols].mean(axis=1, skipna=True)
    df_clean["mean_dbp"] = df_clean[di_cols].mean(axis=1, skipna=True)

    # 6. Told High BP (BPQ020: 1 -> 1, 2 -> 0, recode 7/9 as missing)
    df_clean["high_bp"] = df_clean["BPQ020"].apply(recode_binary_79_as_nan)

    # 7. Told High Cholesterol (BPQ080: 1 -> 1, 2 -> 0, recode 7/9 as missing)
    df_clean["high_chol"] = df_clean["BPQ080"].apply(recode_binary_79_as_nan)

    # 8. Smoker (SMQ020: 1 -> 1, 2 -> 0, recode 7/9 as missing)
    df_clean["smoker"] = df_clean["SMQ020"].apply(recode_binary_79_as_nan)

    # 9. Physical activity:
    # 1 if any of PAQ605/620/635/650/665 == 1.
    # 0 if at least one valid answer (2) exists and none equals 1.
    # NaN if ALL are NaN (or only 7/9).
    pa_cols = [col for col in ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"] if col in df_clean.columns]
    df_clean["phys_activity"] = compute_phys_activity(df_clean, pa_cols)

    # 10. Family history of diabetes (MCQ300C: 1 -> 1, 2 -> 0, recode 7/9 as missing)
    df_clean["family_history"] = df_clean["MCQ300C"].apply(recode_binary_79_as_nan)

    # Select final columns
    final_cols = [
        "SEQN", "cycle", "age", "age_bucket", "sex", "bmi",
        "mean_sbp", "mean_dbp", "high_bp", "high_chol",
        "smoker", "phys_activity", "family_history", "diabetes"
    ]
    return df_clean[final_cols].reset_index(drop=True)


def process_brfss(raw_file: Path) -> pd.DataFrame:
    """Read, filter, harmonize BRFSS 2015 dataset for external validation."""
    print("\n--- [Step 3] Processing BRFSS Data ---")
    df = pd.read_csv(raw_file)
    print(f"  Raw BRFSS shape: {df.shape}")

    # Keep Diabetes_012 classes 0 and 2 only, map 2 -> 1, drop class 1 (prediabetes)
    valid_classes = df["Diabetes_012"].isin([0.0, 2.0])
    df_clean = df[valid_classes].copy()
    df_clean["diabetes"] = (df_clean["Diabetes_012"] == 2.0).astype(int)

    # Map shared features
    rename_cols = {
        "Age": "age_bucket",
        "Sex": "sex",
        "BMI": "bmi",
        "HighBP": "high_bp",
        "HighChol": "high_chol",
        "Smoker": "smoker",
        "PhysActivity": "phys_activity"
    }

    shared_cols = list(rename_cols.keys()) + ["diabetes"]
    df_final = df_clean[shared_cols].rename(columns=rename_cols)
    return df_final.reset_index(drop=True)


def write_data_dictionary(
    nhanes_df: pd.DataFrame,
    brfss_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate comprehensive data dictionary markdown table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nhanes_n = len(nhanes_df)
    brfss_n = len(brfss_df)

    content = f"""# LifeMap AI Diabetes Risk Module - Data Dictionary & Harmonization

## 1. Overview & Cohort Summary
This document specifies the variable mappings, coding rules, missingness patterns, and class distributions for the NHANES (training & internal validation) and BRFSS (external validation) cohorts.

- **NHANES Cohort**: Continuous survey cycles 2013-2014 (_H), 2015-2016 (_I), 2017-2018 (_J). Adult participants aged $\\ge 18$.
  - Total Cleaned Cohort Size: **{nhanes_n:,}**
  - Diabetes Prevalence: **{nhanes_df['diabetes'].mean() * 100:.2f}%** ({nhanes_df['diabetes'].sum():,} cases / {nhanes_n - nhanes_df['diabetes'].sum():,} controls)
- **BRFSS Cohort**: 2015 Behavioral Risk Factor Surveillance System (BRFSS) health indicators.
  - Total Cleaned Cohort Size: **{brfss_n:,}**
  - Diabetes Prevalence: **{brfss_df['diabetes'].mean() * 100:.2f}%** ({brfss_df['diabetes'].sum():,} cases / {brfss_n - brfss_df['diabetes'].sum():,} controls)

---

## 2. Feature Definitions & Cross-Dataset Harmonization Mapping

| Feature Name | Type | Description | NHANES Source & Coding | BRFSS Source & Coding | Shared for Ext. Val? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `diabetes` (Target) | Binary | Diagnosed diabetes status (excluding prediabetes/borderline) | `DIQ010`: 1$\\to$1, 2$\\to$0. Excluded: 3 (borderline), 7, 9, NaN | `Diabetes_012`: 2$\\to$1, 0$\\to$0. Excluded: 1 (prediabetes) | Target |
| `age` | Continuous | Age in years at screening | `RIDAGEYR` ($\\ge 18$) | *Not available continuously* | No (NHANES only) |
| `age_bucket` | Ordinal | Age category (1 to 13) | Derived from `RIDAGEYR` using BRFSS 5-year intervals | `Age`: 1 (18-24) to 13 (80+) | **Yes** |
| `sex` | Binary | Biological sex | `RIAGENDR`: 1 (Male)$\\to$1, 2 (Female)$\\to$0 | `Sex`: 1 (Male), 0 (Female) | **Yes** |
| `bmi` | Continuous | Body Mass Index ($kg/m^2$) | `BMXBMI` | `BMI` | **Yes** |
| `mean_sbp` | Continuous | Mean systolic blood pressure ($mmHg$) | Average of valid readings from `BPXSY1`, `BPXSY2`, `BPXSY3`, `BPXSY4` | *Not measured* | No (NHANES only) |
| `mean_dbp` | Continuous | Mean diastolic blood pressure ($mmHg$) | Average of valid readings from `BPXDI1`, `BPXDI2`, `BPXDI3`, `BPXDI4` | *Not measured* | No (NHANES only) |
| `high_bp` | Binary | Ever told high blood pressure | `BPQ020`: 1$\\to$1, 2$\\to$0; 7, 9$\\to$NaN | `HighBP`: 1$\\to$1, 0$\\to$0 | **Yes** |
| `high_chol` | Binary | Ever told high cholesterol | `BPQ080`: 1$\\to$1, 2$\\to$0; 7, 9$\\to$NaN | `HighChol`: 1$\\to$1, 0$\\to$0 | **Yes** |
| `smoker` | Binary | Lifetime cigarette smoker ($\\ge 100$ cigs) | `SMQ020`: 1$\\to$1, 2$\\to$0; 7, 9$\\to$NaN | `Smoker`: 1$\\to$1, 0$\\to$0 | **Yes** |
| `phys_activity` | Binary | Physical activity engagement | 1 if any `PAQ605-665`==1; 0 if at least one valid answer and none==1; NaN if all missing or 7/9 | `PhysActivity`: 1$\\to$1, 0$\\to$0 | **Yes** |
| `family_history` | Binary | Close biological relative had diabetes | `MCQ300C`: 1$\\to$1, 2$\\to$0; 7, 9$\\to$NaN | *Not measured in this subset* | No (NHANES only) |

---

## 3. Age Interval Reference (BRFSS 13 Buckets)

- `1`: 18-24 years
- `2`: 25-29 years
- `3`: 30-34 years
- `4`: 35-39 years
- `5`: 40-44 years
- `6`: 45-49 years
- `7`: 50-54 years
- `8`: 55-59 years
- `9`: 60-64 years
- `10`: 65-69 years
- `11`: 70-74 years
- `12`: 75-79 years
- `13`: 80+ years

---

## 4. Missingness Analysis

### NHANES Cohort (N = {nhanes_n:,})
| Column | Missing Count | Missing Percentage | Imputation Strategy in Phase 2 |
| :--- | :---: | :---: | :--- |
"""

    for col in ["age", "age_bucket", "sex", "bmi", "mean_sbp", "mean_dbp", "high_bp", "high_chol", "smoker", "phys_activity", "family_history", "diabetes"]:
        miss_cnt = nhanes_df[col].isna().sum()
        miss_pct = (miss_cnt / nhanes_n) * 100
        strategy = "None (Complete)" if miss_cnt == 0 else "Median (Continuous Pipeline)" if "mean" in col or col == "bmi" else "Mode / SimpleImputer (Binary Pipeline)"
        content += f"| `{col}` | {miss_cnt:,} | {miss_pct:.2f}% | {strategy} |\n"

    content += f"""
### BRFSS Cohort (N = {brfss_n:,})
| Column | Missing Count | Missing Percentage | Imputation Strategy |
| :--- | :---: | :---: | :--- |
"""
    for col in brfss_df.columns:
        miss_cnt = brfss_df[col].isna().sum()
        miss_pct = (miss_cnt / brfss_n) * 100
        content += f"| `{col}` | {miss_cnt:,} | {miss_pct:.2f}% | None (Clean Complete Set) |\n"

    content += f"""
---

## 5. Class Balance & Distribution

| Metric | NHANES Cohort | BRFSS Cohort |
| :--- | :---: | :---: |
| Total Adults | {nhanes_n:,} | {brfss_n:,} |
| Diabetes Negative (0) | {nhanes_df['diabetes'].value_counts()[0]:,} ({nhanes_df['diabetes'].value_counts(normalize=True)[0] * 100:.2f}%) | {brfss_df['diabetes'].value_counts()[0]:,} ({brfss_df['diabetes'].value_counts(normalize=True)[0] * 100:.2f}%) |
| Diabetes Positive (1) | {nhanes_df['diabetes'].value_counts()[1]:,} ({nhanes_df['diabetes'].value_counts(normalize=True)[1] * 100:.2f}%) | {brfss_df['diabetes'].value_counts()[1]:,} ({brfss_df['diabetes'].value_counts(normalize=True)[1] * 100:.2f}%) |
| Imbalance Ratio (Neg:Pos) | {nhanes_df['diabetes'].value_counts()[0] / nhanes_df['diabetes'].value_counts()[1]:.2f} : 1 | {brfss_df['diabetes'].value_counts()[0] / brfss_df['diabetes'].value_counts()[1]:.2f} : 1 |

*Note: In accordance with capstone leakage precautions, no laboratory values (fasting plasma glucose, oral glucose tolerance test, HbA1c, insulin) were used as features.*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[Saved] Data dictionary written to: {output_path}")


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    raw_nhanes_dir = root_dir / "data" / "raw" / "nhanes"
    raw_brfss_file = root_dir / "data" / "raw" / "brfss" / "diabetes_012_health_indicators_BRFSS2015.csv"
    processed_dir = root_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify NHANES files
    verify_nhanes_files(raw_nhanes_dir)

    # 2. Process NHANES
    nhanes_df = process_nhanes(raw_nhanes_dir)
    nhanes_out = processed_dir / "nhanes_diabetes_processed.csv"
    nhanes_df.to_csv(nhanes_out, index=False)
    print(f"\n[Saved] Processed NHANES saved to: {nhanes_out}")

    # 3. Process BRFSS
    brfss_df = process_brfss(raw_file=raw_brfss_file)
    brfss_out = processed_dir / "brfss_diabetes_processed.csv"
    brfss_df.to_csv(brfss_out, index=False)
    print(f"[Saved] Processed BRFSS saved to: {brfss_out}")

    # 4. Write Data Dictionary
    data_dict_path = root_dir / "diabetes_module" / "reports" / "tables" / "data_dictionary.md"
    write_data_dictionary(nhanes_df, brfss_df, data_dict_path)

    # 5. Checkpoint output & Detailed Missingness
    print("\n" + "=" * 70)
    print("                    UPDATED PREPROCESSING CHECKPOINT")
    print("=" * 70)

    feature_cols_nhanes = [
        "age", "age_bucket", "sex", "bmi", "mean_sbp", "mean_dbp",
        "high_bp", "high_chol", "smoker", "phys_activity", "family_history", "diabetes"
    ]

    print("\n[NHANES Cohort - Missing Value Counts & Percentages]")
    print(f"{'Feature':<18} | {'Missing Count':<14} | {'Missing Percentage':<18} | {'Total':<10}")
    print("-" * 70)
    for col in feature_cols_nhanes:
        m_count = nhanes_df[col].isna().sum()
        m_pct = (m_count / len(nhanes_df)) * 100
        print(f"{col:<18} | {m_count:<14} | {m_pct:>6.2f}%            | {len(nhanes_df):<10}")

    print("\n[BRFSS Cohort (Shared Features) - Missing Value Counts & Percentages]")
    print(f"{'Feature':<18} | {'Missing Count':<14} | {'Missing Percentage':<18} | {'Total':<10}")
    print("-" * 70)
    for col in brfss_df.columns:
        m_count = brfss_df[col].isna().sum()
        m_pct = (m_count / len(brfss_df)) * 100
        print(f"{col:<18} | {m_count:<14} | {m_pct:>6.2f}%            | {len(brfss_df):<10}")

    print("\n" + "-" * 70)
    print(f"NHANES Processed Shape: {nhanes_df.shape}")
    print("NHANES Class Balance:")
    print(nhanes_df["diabetes"].value_counts(dropna=False).to_string())
    print("\nNHANES Class Proportions:")
    print(nhanes_df["diabetes"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_string())
    print("-" * 70)
    print(f"BRFSS Processed Shape:  {brfss_df.shape}")
    print("BRFSS Class Balance:")
    print(brfss_df["diabetes"].value_counts(dropna=False).to_string())
    print("\nBRFSS Class Proportions:")
    print(brfss_df["diabetes"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_string())
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
