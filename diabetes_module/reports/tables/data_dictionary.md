# LifeMap AI Diabetes Risk Module - Data Dictionary & Harmonization

## 1. Overview & Cohort Summary
This document specifies the variable mappings, coding rules, missingness patterns, and class distributions for the NHANES (training & internal validation) and BRFSS (external validation) cohorts.

- **NHANES Cohort**: Continuous survey cycles 2013-2014 (_H), 2015-2016 (_I), 2017-2018 (_J). Adult participants aged $\ge 18$.
  - Total Cleaned Cohort Size: **17,468**
  - Diabetes Prevalence: **14.02%** (2,449 cases / 15,019 controls)
- **BRFSS Cohort**: 2015 Behavioral Risk Factor Surveillance System (BRFSS) health indicators.
  - Total Cleaned Cohort Size: **249,049**
  - Diabetes Prevalence: **14.19%** (35,346 cases / 213,703 controls)

---

## 2. Feature Definitions & Cross-Dataset Harmonization Mapping

| Feature Name | Type | Description | NHANES Source & Coding | BRFSS Source & Coding | Shared for Ext. Val? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `diabetes` (Target) | Binary | Diagnosed diabetes status (excluding prediabetes/borderline) | `DIQ010`: 1$\to$1, 2$\to$0. Excluded: 3 (borderline), 7, 9, NaN | `Diabetes_012`: 2$\to$1, 0$\to$0. Excluded: 1 (prediabetes) | Target |
| `age` | Continuous | Age in years at screening | `RIDAGEYR` ($\ge 18$) | *Not available continuously* | No (NHANES only) |
| `age_bucket` | Ordinal | Age category (1 to 13) | Derived from `RIDAGEYR` using BRFSS 5-year intervals | `Age`: 1 (18-24) to 13 (80+) | **Yes** |
| `sex` | Binary | Biological sex | `RIAGENDR`: 1 (Male)$\to$1, 2 (Female)$\to$0 | `Sex`: 1 (Male), 0 (Female) | **Yes** |
| `bmi` | Continuous | Body Mass Index ($kg/m^2$) | `BMXBMI` | `BMI` | **Yes** |
| `mean_sbp` | Continuous | Mean systolic blood pressure ($mmHg$) | Average of valid readings from `BPXSY1`, `BPXSY2`, `BPXSY3`, `BPXSY4` | *Not measured* | No (NHANES only) |
| `mean_dbp` | Continuous | Mean diastolic blood pressure ($mmHg$) | Average of valid readings from `BPXDI1`, `BPXDI2`, `BPXDI3`, `BPXDI4` | *Not measured* | No (NHANES only) |
| `high_bp` | Binary | Ever told high blood pressure | `BPQ020`: 1$\to$1, 2$\to$0; 7, 9$\to$NaN | `HighBP`: 1$\to$1, 0$\to$0 | **Yes** |
| `high_chol` | Binary | Ever told high cholesterol | `BPQ080`: 1$\to$1, 2$\to$0; 7, 9$\to$NaN | `HighChol`: 1$\to$1, 0$\to$0 | **Yes** |
| `smoker` | Binary | Lifetime cigarette smoker ($\ge 100$ cigs) | `SMQ020`: 1$\to$1, 2$\to$0; 7, 9$\to$NaN | `Smoker`: 1$\to$1, 0$\to$0 | **Yes** |
| `phys_activity` | Binary | Physical activity engagement | 1 if any `PAQ605-665`==1; 0 if at least one valid answer and none==1; NaN if all missing or 7/9 | `PhysActivity`: 1$\to$1, 0$\to$0 | **Yes** |
| `family_history` | Binary | Close biological relative had diabetes | `MCQ300C`: 1$\to$1, 2$\to$0; 7, 9$\to$NaN | *Not measured in this subset* | No (NHANES only) |

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

### NHANES Cohort (N = 17,468)
| Column | Missing Count | Missing Percentage | Imputation Strategy in Phase 2 |
| :--- | :---: | :---: | :--- |
| `age` | 0 | 0.00% | None (Complete) |
| `age_bucket` | 0 | 0.00% | None (Complete) |
| `sex` | 0 | 0.00% | None (Complete) |
| `bmi` | 993 | 5.68% | Median (Continuous Pipeline) |
| `mean_sbp` | 1,385 | 7.93% | Median (Continuous Pipeline) |
| `mean_dbp` | 1,385 | 7.93% | Median (Continuous Pipeline) |
| `high_bp` | 21 | 0.12% | Mode / SimpleImputer (Binary Pipeline) |
| `high_chol` | 106 | 0.61% | Mode / SimpleImputer (Binary Pipeline) |
| `smoker` | 11 | 0.06% | Mode / SimpleImputer (Binary Pipeline) |
| `phys_activity` | 0 | 0.00% | None (Complete) |
| `family_history` | 1,196 | 6.85% | Mode / SimpleImputer (Binary Pipeline) |
| `diabetes` | 0 | 0.00% | None (Complete) |

### BRFSS Cohort (N = 249,049)
| Column | Missing Count | Missing Percentage | Imputation Strategy |
| :--- | :---: | :---: | :--- |
| `age_bucket` | 0 | 0.00% | None (Clean Complete Set) |
| `sex` | 0 | 0.00% | None (Clean Complete Set) |
| `bmi` | 0 | 0.00% | None (Clean Complete Set) |
| `high_bp` | 0 | 0.00% | None (Clean Complete Set) |
| `high_chol` | 0 | 0.00% | None (Clean Complete Set) |
| `smoker` | 0 | 0.00% | None (Clean Complete Set) |
| `phys_activity` | 0 | 0.00% | None (Clean Complete Set) |
| `diabetes` | 0 | 0.00% | None (Clean Complete Set) |

---

## 5. Class Balance & Distribution

| Metric | NHANES Cohort | BRFSS Cohort |
| :--- | :---: | :---: |
| Total Adults | 17,468 | 249,049 |
| Diabetes Negative (0) | 15,019 (85.98%) | 213,703 (85.81%) |
| Diabetes Positive (1) | 2,449 (14.02%) | 35,346 (14.19%) |
| Imbalance Ratio (Neg:Pos) | 6.13 : 1 | 6.05 : 1 |

*Note: In accordance with capstone leakage precautions, no laboratory values (fasting plasma glucose, oral glucose tolerance test, HbA1c, insulin) were used as features.*
