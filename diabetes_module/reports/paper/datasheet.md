# LifeMap AI Diabetes Module - Datasheet for Datasets

Following the standard Datasheets for Datasets framework (Gebru et al., 2021).

---

## 1. National Health and Nutrition Examination Survey (NHANES)

### Motivation
- **Purpose:** Assess the health and nutritional status of adults and children in the United States through interviews and physical examinations.
- **Funding & Oversight:** National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC).

### Composition & Sampling
- **Study Cycles Used:** 2013-2014 (Cycle H), 2015-2016 (Cycle I), 2017-2018 (Cycle J).
- **Sampling Frame:** Complex, stratified, multistage probability sample of the civilian, non-institutionalized US population.
- **Raw Cohort Size:** 29,400 total participants across cycles H (10,175), I (9,971), and J (9,254).
- **Adult Cohort (>= 18 years):** 17,961 participants (Cycle H: 6,113; Cycle I: 5,992; Cycle J: 5,856).
- **Final Harmonized Sample:** 17,468 participants after excluding indeterminate responses:
  - Excluded: Under 18 years ($n=11,439$), borderline/prediabetes ($n=482$), refused/don't know ($n=11$).
  - Class Distribution: 2,449 Diabetic (14.02%), 15,019 Non-Diabetic (85.98%).
  - Class Imbalance: 1:6.13.

### Collection Process
- **Data Collection Modality:** Two-part protocol:
  1. In-home face-to-face questionnaire interviews (Demographics, Medical Conditions, Physical Activity, Smoking).
  2. Mobile Examination Center (MEC) visit: Physical body measurements (BMI), multiple seated blood pressure determinations, and laboratory phlebotomy assays (glycated hemoglobin, HbA1c).
- **Timeframe:** Data collected continuously from January 2013 through December 2018.

### Preprocessing & Data Cleaning
- **Target Label Definition:** Derived from questionnaire variable `DIQ010` ("Have you ever been told by a doctor or health professional that you have diabetes?"):
  - `1.0` (Yes) $\rightarrow$ Diabetic ($y=1$).
  - `2.0` (No) $\rightarrow$ Non-Diabetic ($y=0$).
  - `3.0` (Borderline), `7.0` (Refused), `9.0` (Don't know) $\rightarrow$ Dropped.
- **Harmonization & Imputation:**
  - Continuous blood pressures: Mean calculated across valid readings among `BPXSY1-4` and `BPXDI1-4`.
  - Imputation: Median for continuous variables, mode for binary variables, fitted strictly on the training partition ($n=12,227$) to prevent data leakage.
  - HbA1c: Laboratory assay from MEC phlebotomy; missing in 8.77% of adults ($n=1,532$) due to MEC non-attendance.

### Biases & Ethical Considerations
- **Recall Bias:** Self-reported diabetes diagnoses omit undiagnosed asymptomatic type 2 diabetes.
- **Survivor & Healthy Volunteer Bias:** Institutionalized individuals, hospitalized patients, and active military personnel are excluded from the sampling frame.
- **Terms of Use & Licensing:** Public-domain US Federal Government data; users should check the CDC terms of use (https://www.cdc.gov/nchs/nhanes/).

---

## 2. Behavioral Risk Factor Surveillance System (BRFSS) 2015

### Motivation
- **Purpose:** Monitor state-level health-related risk behaviors, chronic health conditions, and use of preventive services across all 50 US states and territories.
- **Funding & Oversight:** Centers for Disease Control and Prevention (CDC) Division of Population Health.

### Composition & Sampling
- **Survey Year:** 2015 annual cross-sectional survey.
- **Sampling Frame:** Landline and cellular telephone random digit dialing (RDD) covering adult civilian residents.
- **Raw Survey Size:** 253,680 records in the curated indicator dataset.
- **Filtering & Funnel:**
  - Excluded: Prediabetes (Class 1, $n=4,631$) to align with binary clinical screening.
  - Final Evaluated Cohort: 249,049 adult respondents.
  - Class Distribution: 35,346 Diabetic (14.19%), 213,703 Non-Diabetic (85.81%).
  - Class Imbalance: 1:6.05.

### Harmonization & Usage
- **Features Shared with NHANES (Shared-7):**
  1. `Age`: Harmonized into 13 ordinal five-year age intervals (18-24 up to 80+).
  2. `Sex`: Binary (Male=1, Female=0).
  3. `BMI`: Continuous body mass index calculated from self-reported height and weight.
  4. `HighBP`: Self-reported hypertension history.
  5. `HighChol`: Self-reported hypercholesterolemia history.
  6. `Smoker`: Lifetime consumption of $\ge 100$ cigarettes.
  7. `PhysActivity`: Any physical activity in the past 30 days outside regular job.
- **Validation Protocol:** Zero-shot external transfer testing. The models were trained and tuned on NHANES and evaluated on BRFSS without any parameter fine-tuning, threshold re-optimization, or domain adaptation.

### Biases & Ethical Considerations
- **Telephone Non-Response Bias:** Landline and mobile telephone non-response has increased over survey years.
- **Self-Report Measurement Error:** Height and weight are self-reported via telephone, typically underestimating true BMI relative to physical clinical examination.
- **Terms of Use & Licensing:** CDC public health data; users should check the CDC terms of use (https://www.cdc.gov/brfss/).
