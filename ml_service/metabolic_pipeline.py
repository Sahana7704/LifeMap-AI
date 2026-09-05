"""
LifeMap AI — Metabolic Risk Module: Diabetes + Obesity
Implements:
  STEP 1: Extraction (deterministic, no reasoning)
  STEP 2: Pre-generation gate (CODE — hard stop, not optional, code-only BMI calculation)
  STEP 3: Risk scoring (grounded in ADA + WHO/ICMR-NIN Asian BMI cutoffs)
  STEP 4: Post-generation audit (CODE)
"""

import os
import re
import json
import logging
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("lifemap_metabolic")

# =====================================================================
# SYSTEM PROMPTS (Verbatim as specified in user specification)
# =====================================================================

METABOLIC_STEP1_SYSTEM_PROMPT = (
    "You are a metabolic-panel extractor. Output ONLY fields explicitly "
    "present in the OCR text — never infer, estimate, calculate, or default "
    "a value, even if two other values would let you derive it (e.g. do NOT "
    "calculate BMI from height+weight unless BMI itself is explicitly printed "
    "in the source — extraction never calculates, only reads).\n\n"
    "Rules:\n"
    "1. Extract ONLY: Fasting Blood Glucose, HbA1c, Random Blood Glucose, "
    "   Height, Weight, BMI, Waist Circumference, Age, Sex — and ONLY if each "
    "   is explicitly printed in the source.\n"
    "2. For each field: value, unit, reference_range (as printed), "
    "   source_flagged (true if bolded/marked abnormal in source).\n"
    "3. Missing field -> null. Never backfill or estimate.\n"
    "4. Do not extract or invent smoking status, cardiac history, or any "
    "   field not listed above.\n"
    "5. Output strict JSON only, no prose.\n\n"
    "Schema:\n"
    "{\n"
    '  "fasting_glucose": {"value": number|null, "unit": str, "reference_range": str, "source_flagged": bool},\n'
    '  "hba1c": {"value": number|null, "unit": str, "reference_range": str, "source_flagged": bool},\n'
    '  "random_glucose": {"value": number|null, "unit": str, "reference_range": str, "source_flagged": bool},\n'
    '  "height_cm": number|null,\n'
    '  "weight_kg": number|null,\n'
    '  "bmi": {"value": number|null, "unit": str, "reference_range": str, "source_flagged": bool, "source": "printed"|"not_present"},\n'
    '  "waist_circumference_cm": number|null,\n'
    '  "age": number|null,\n'
    '  "sex": str|null,\n'
    '  "extraction_confidence": "high"|"low",\n'
    '  "raw_text_char_count": number\n'
    "}"
)

METABOLIC_STEP3_SYSTEM_PROMPT = (
    "You are a metabolic risk classifier. You receive ONLY the verified JSON "
    "from Step 2 — no access to the original document.\n\n"
    "DIABETES classification (ADA criteria, apply exactly):\n"
    "- Fasting Glucose >=126 mg/dL -> 'Diagnostic range for diabetes'\n"
    "- Fasting Glucose 100–125 -> 'Prediabetes range'\n"
    "- Fasting Glucose <100 -> 'Normal range'\n"
    "- HbA1c >=6.5% -> 'Diagnostic range for diabetes'\n"
    "- HbA1c 5.7–6.4% -> 'Prediabetes range'\n"
    "- HbA1c <5.7% -> 'Normal range'\n"
    "- If glucose and HbA1c disagree, report both separately — never average.\n\n"
    "OBESITY classification (WHO Asian-adjusted cutoffs — state which "
    "standard you're using in the output):\n"
    "- BMI <18.5 -> Underweight\n"
    "- BMI 18.5–22.9 -> Normal\n"
    "- BMI 23–24.9 -> Overweight (Asian cutoff; standard WHO cutoff is 25–29.9)\n"
    "- BMI >=25 -> Obese (Asian cutoff; standard WHO cutoff is >=30)\n"
    "- If waist_circumference_cm present: >90cm (men) or >80cm (women) -> "
    "  flag central obesity as additional risk modifier.\n\n"
    "COMBINED risk note: state explicitly (in one line) how the obesity "
    "classification modifies diabetes risk interpretation — e.g. obesity + "
    "prediabetes-range glucose = elevated concern — but do NOT invent a new "
    "blended numeric score. Report both classifications and their "
    "relationship in plain language, not a fabricated single percentage.\n\n"
    "Hard rules:\n"
    "1. Reference ONLY non-null fields from the input. If a field is null, "
    "   never mention it, positively or negatively.\n"
    "2. Every source_flagged=true field MUST appear in output — checklist "
    "   this before finalizing.\n"
    "3. Never soften a diagnostic-range result into casual 'manageable' "
    "   language without also stating its clinical significance plainly.\n"
    "4. Field-count & partial data rules:\n"
    "   - Total available fields include all verified fields: fasting_glucose, "
    "     hba1c, random_glucose, bmi, height_cm, weight_kg, waist_circumference_cm.\n"
    "   - If total verified fields >= 2 (e.g., when anthropometrics BMI/height/weight/waist "
    "     are present): do NOT claim the report has fewer than two fields, and do NOT claim "
    "     assessment is data-limited. Classify the condition with data confidently (e.g. Obesity), "
    "     and explicitly state that the other condition (e.g. Diabetes) cannot be assessed "
    "     because its panel is pending.\n"
    "   - Only if total verified fields < 2, state that assessment is data-limited.\n\n"
    "Output classifications and explanation now."
)

# =====================================================================
# LLM CALLER DISPATCHER
# =====================================================================

def call_metabolic_llm(system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    """Attempts to invoke Gemini or OpenAI if configured in the environment."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={"response_mime_type": "application/json"}
            )
            prompt = f"{system_prompt}\n\nUser Input:\n{user_prompt}"
            response = model.generate_content(prompt)
            if response and response.text:
                cleaned = response.text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                return json.loads(cleaned.strip())
        except Exception as e:
            logger.warning(f"Metabolic Gemini API warning: {e}")

    if openai_key:
        try:
            import urllib.request
            req_data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(req_data).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            logger.warning(f"Metabolic OpenAI API warning: {e}")

    return None

# =====================================================================
# STEP 1: EXTRACTION (Deterministic, No Reasoning, No Derivations)
# =====================================================================

def extract_field_robust(
    text: str,
    header_patterns: List[str],
    min_val: float,
    max_val: float,
    unit_pattern: Optional[str] = None,
    allow_mmol: bool = False
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[str]]:
    """
    Robust multi-section extractor:
    Scans entire document text for each test header independently.
    Handles intervening lines such as 'Method: HPLC', 'Sample: Serum', etc.
    Avoids capturing reference intervals (< 5.7, 70-100) or subsequent test sections.
    Returns (extracted_value, start_idx, end_idx, unit_detected).
    """
    combined_header = r'(?:' + '|'.join(header_patterns) + r')\b'
    for m in re.finditer(combined_header, text, re.I):
        start_pos = m.end()
        snippet = text[start_pos:start_pos + 450]
        m_stop = re.search(r'(?:biological\s+reference|reference\s+interval|reference\s+range|ref\.\s*range|ref\s*interval|biological\s*ref)', snippet, re.I)
        search_window = snippet[:m_stop.start()] if m_stop else snippet

        # Check for mmol/L unit context in search_window
        window_has_mmol = bool(re.search(r'\bmmol\s*[\/\.]\s*[lL]\b', search_window, re.I))

        # Priority 1: Explicit 'Result: X' or 'Value: X'
        m_res = re.search(r'(?:result|value|observed|reading|level)[\s:\-|=]*([0-9]{1,3}(?:\.[0-9]+)?)', search_window, re.I)
        if m_res:
            try:
                val = float(m_res.group(1))
                if min_val <= val <= max_val:
                    return val, m.start(), start_pos + m_res.end(), "mg/dL" if not window_has_mmol else "mmol/L"
                if allow_mmol and window_has_mmol and 2.0 <= val <= 35.0:
                    return val, m.start(), start_pos + m_res.end(), "mmol/L"
            except ValueError:
                pass

        # Priority 2: Number followed by matching unit (or flag then unit, like 118 H mg/dL)
        if unit_pattern:
            m_unit = re.search(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:[A-Za-z]\s+)?(' + unit_pattern + r')', search_window, re.I)
            if m_unit:
                try:
                    val = float(m_unit.group(1))
                    unit_str = m_unit.group(2)
                    is_mmol = "mmol" in unit_str.lower()
                    if min_val <= val <= max_val:
                        return val, m.start(), start_pos + m_unit.end(), "mmol/L" if is_mmol else "mg/dL"
                    if allow_mmol and is_mmol and 2.0 <= val <= 35.0:
                        return val, m.start(), start_pos + m_unit.end(), "mmol/L"
                except ValueError:
                    pass

        # Priority 3: Standalone number in lines that do not describe methods/instruments
        lines = search_window.split('\n')
        cleaned_lines = []
        for line in lines[:3]:
            if re.match(r'^\s*(?:method|specimen|sample|instrument|analyzer|technology)\s*:', line, re.I):
                continue
            if re.search(r'\b(?:expected|interpretation|comment|note|eag|estimated)\b', line, re.I):
                continue
            cleaned_lines.append(line)
        filtered_window = '\n'.join(cleaned_lines)

        for m_num in re.finditer(r'(?:([<>=]+)\s*)?([0-9]{1,3}(?:\.[0-9]+)?)\b', filtered_window):
            if m_num.group(1):
                # Preceded by comparison operators like <, >, <=, >= (reference thresholds)
                continue
            try:
                val = float(m_num.group(2))
                if min_val <= val <= max_val:
                    return val, m.start(), start_pos + len(search_window), "mmol/L" if window_has_mmol else "mg/dL"
                if allow_mmol and window_has_mmol and 2.0 <= val <= 35.0:
                    return val, m.start(), start_pos + len(search_window), "mmol/L"
            except ValueError:
                pass

    return None, None, None, None

def step1_metabolic_extraction(raw_ocr_text: str) -> Dict[str, Any]:
    """
    Extract ONLY explicitly printed values: Fasting Glucose, HbA1c, Random Glucose,
    Height, Weight, BMI (only if printed), Waist Circumference, Age, Sex.
    Never calculates BMI from height+weight in this step.
    Scans entire document text independently for all target fields across multi-section formats.
    """
    normalized_text = unicodedata.normalize('NFKC', raw_ocr_text).replace('：', ':').replace('，', ',')
    char_count = len(normalized_text)

    # Try external LLM call first if available
    llm_result = call_metabolic_llm(
        system_prompt=METABOLIC_STEP1_SYSTEM_PROMPT,
        user_prompt=f"OCR Text:\n{normalized_text}"
    )
    if llm_result and isinstance(llm_result, dict) and "fasting_glucose" in llm_result:
        return llm_result

    # High-precision deterministic extractor strictly following Step 1 rules
    has_bold_notice = bool(re.search(r'(?:highlighted|bold|abnormal|\*)', normalized_text, re.I))

    # Fasting Glucose (Scans independently across document, wrapped in try/except for Bug K)
    fasting_glucose = None
    try:
        fg_val, fg_start, fg_end, fg_unit = extract_field_robust(
            normalized_text,
            [r'fasting\s*(?:plasma\s*|blood\s*)?(?:glucose|sugar)', r'fbs\b', r'fbg\b', r'glucose\s*\(?fasting\)?', r'blood\s*sugar\s*\(?fasting\)?'],
            40.0, 500.0,
            unit_pattern=r'(?:mg\s*[\/\.]\s*d[lL]|mmol\s*[\/\.]\s*[lL])',
            allow_mmol=True
        )
        if fg_val is None:
            # Fallback to generic glucose if not marked as random or post-prandial
            if not re.search(r'\b(?:random|post\s*prandial|ppbs|rbs)\b', normalized_text, re.I):
                cleaned_text = re.sub(r'estimated\s+avg(?:erage)?\s+glucose.*?(?=\n\n|\Z)', '', normalized_text, flags=re.I | re.S)
                cleaned_text = re.sub(r'glucose\s*\(?eag\)?.*?(?=\n\n|\Z)', '', cleaned_text, flags=re.I | re.S)
                fg_val, fg_start, fg_end, fg_unit = extract_field_robust(
                    cleaned_text,
                    [r'blood\s*(?:glucose|sugar)', r'\bglucose\b'],
                    40.0, 500.0,
                    unit_pattern=r'(?:mg\s*[\/\.]\s*d[lL]|mmol\s*[\/\.]\s*[lL])',
                    allow_mmol=True
                )
        if fg_val is not None:
            ctx = normalized_text[max(0, (fg_start or 0) - 30):min(len(normalized_text), (fg_end or 0) + 30)].lower()
            if fg_unit == "mmol/L":
                converted_val = round(fg_val * 18.0182, 1)
                flagged = (fg_val >= 5.6) or ("high" in ctx) or ("abnormal" in ctx) or has_bold_notice
                fasting_glucose = {
                    "value": converted_val,
                    "unit": "mg/dL",
                    "original_value": fg_val,
                    "original_unit": "mmol/L",
                    "reference_range": "3.9 - 5.5 mmol/L (70 - 99 mg/dL)",
                    "source_flagged": flagged
                }
            else:
                flagged = (fg_val >= 100.0) or ("high" in ctx) or ("abnormal" in ctx) or has_bold_notice
                fasting_glucose = {
                    "value": fg_val,
                    "unit": "mg/dL",
                    "reference_range": "70 - 99 mg/dL",
                    "source_flagged": flagged
                }
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting fasting_glucose: {e}")

    # HbA1c (Scans independently across document, wrapped in try/except for Bug K)
    hba1c = None
    try:
        hba1c_val, h_start, h_end, _ = extract_field_robust(
            normalized_text,
            [r'hba1c', r'hb\s*[-–]?\s*a1c', r'a1c', r'(?:glycated|glycosylated)\s*(?:ha?emoglobin|hb)'],
            3.5, 20.0,
            unit_pattern=r'(?:%|percent)'
        )
        if hba1c_val is not None:
            ctx = normalized_text[max(0, (h_start or 0) - 30):min(len(normalized_text), (h_end or 0) + 30)].lower()
            flagged = (hba1c_val >= 5.7) or ("high" in ctx) or ("abnormal" in ctx) or has_bold_notice
            hba1c = {
                "value": hba1c_val,
                "unit": "%",
                "reference_range": "< 5.7%",
                "source_flagged": flagged
            }
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting hba1c: {e}")

    # Random Glucose (Scans independently across document, wrapped in try/except for Bug K)
    random_glucose = None
    try:
        rg_val, rg_start, rg_end, rg_unit = extract_field_robust(
            normalized_text,
            [r'random\s*(?:blood\s*)?(?:glucose|sugar)', r'post\s*prandial\s*(?:blood\s*)?(?:glucose|sugar)', r'ppbs', r'rbs', r'glucose\s*\(?pp\)?'],
            40.0, 500.0,
            unit_pattern=r'(?:mg\s*[\/\.]\s*d[lL]|mmol\s*[\/\.]\s*[lL])',
            allow_mmol=True
        )
        if rg_val is not None:
            ctx = normalized_text[max(0, (rg_start or 0) - 30):min(len(normalized_text), (rg_end or 0) + 30)].lower()
            if rg_unit == "mmol/L":
                converted_val = round(rg_val * 18.0182, 1)
                flagged = (rg_val >= 7.8) or ("high" in ctx) or ("abnormal" in ctx) or has_bold_notice
                random_glucose = {
                    "value": converted_val,
                    "unit": "mg/dL",
                    "original_value": rg_val,
                    "original_unit": "mmol/L",
                    "reference_range": "70 - 139 mg/dL",
                    "source_flagged": flagged
                }
            else:
                flagged = (rg_val >= 140.0) or ("high" in ctx) or ("abnormal" in ctx) or has_bold_notice
                random_glucose = {
                    "value": rg_val,
                    "unit": "mg/dL",
                    "reference_range": "70 - 139 mg/dL",
                    "source_flagged": flagged
                }
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting random_glucose: {e}")

    # Height (cm) (Scans independently across document, wrapped in try/except for Bug K)
    height_cm = None
    try:
        m_ht = re.search(r'\b(?:height|ht)\b[\s:\-|=]*([0-9]{1,3}(?:\.[0-9]+)?)\s*(cm|cms|m|mtrs|meters)?\b', normalized_text, re.I)
        if m_ht:
            val = float(m_ht.group(1))
            unit = (m_ht.group(2) or "").lower()
            if "m" in unit and "cm" not in unit:
                if 1.2 <= val <= 2.3:
                    height_cm = round(val * 100.0, 1)
            elif 100.0 <= val <= 230.0:
                height_cm = val
        if height_cm is None:
            ht_val, _, _, _ = extract_field_robust(
                normalized_text,
                [r'\bheight\b', r'\bht\b'],
                100.0, 230.0,
                unit_pattern=r'(?:cm|cms)\b'
            )
            if ht_val is None:
                ht_m, _, _, _ = extract_field_robust(
                    normalized_text,
                    [r'\bheight\b', r'\bht\b'],
                    1.2, 2.3,
                    unit_pattern=r'(?:m|mtrs|meters)\b'
                )
                if ht_m is not None:
                    ht_val = round(ht_m * 100.0, 1)
            if ht_val is not None:
                height_cm = ht_val
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting height_cm: {e}")

    # Weight (kg) (Scans independently across document, wrapped in try/except for Bug K)
    weight_kg = None
    try:
        wt_val, _, _, _ = extract_field_robust(
            normalized_text,
            [r'\bweight\b', r'\bwt\b'],
            30.0, 250.0,
            unit_pattern=r'(?:kg|kgs)\b'
        )
        if wt_val is not None:
            weight_kg = wt_val
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting weight_kg: {e}")

    # BMI (ONLY if explicitly printed in source, never calculated here in Step 1, wrapped in try/except for Bug K)
    bmi = None
    try:
        bmi_val, bmi_start, bmi_end, _ = extract_field_robust(
            normalized_text,
            [r'body\s*mass\s*index', r'\bbmi\b'],
            12.0, 65.0,
            unit_pattern=r'(?:kg\s*[\/\.]\s*m[²2])?'
        )
        if bmi_val is not None:
            ctx = normalized_text[max(0, (bmi_start or 0) - 30):min(len(normalized_text), (bmi_end or 0) + 30)].lower()
            flagged = (bmi_val >= 23.0 or bmi_val < 18.5) or ("high" in ctx) or ("abnormal" in ctx) or has_bold_notice
            bmi = {
                "value": bmi_val,
                "unit": "kg/m²",
                "reference_range": "18.5 - 22.9 kg/m² (Asian cutoff)",
                "source_flagged": flagged,
                "source": "printed"
            }
        else:
            bmi = {
                "value": None,
                "unit": "kg/m²",
                "reference_range": "18.5 - 22.9 kg/m²",
                "source_flagged": False,
                "source": "not_present"
            }
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting bmi: {e}")
        bmi = {
            "value": None,
            "unit": "kg/m²",
            "reference_range": "18.5 - 22.9 kg/m²",
            "source_flagged": False,
            "source": "not_present"
        }

    # Waist Circumference (cm) (Scans independently across document, wrapped in try/except for Bug K)
    waist_circumference_cm = None
    try:
        wc_val, _, _, _ = extract_field_robust(
            normalized_text,
            [r'waist(?:\s*circumference)?', r'\bwc\b'],
            45.0, 180.0,
            unit_pattern=r'(?:cm|cms)\b'
        )
        if wc_val is not None:
            waist_circumference_cm = wc_val
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting waist_circumference_cm: {e}")

    # Age (Scans independently across document, wrapped in try/except for Bug K)
    age = None
    try:
        m_age = re.search(r'(?:age\s*[\/\\]\s*(?:gender|sex)|age)\s*[:\-\|\=\.]*\s*([0-9]{1,3})\s*(?:years?|yrs?|yesrs|y)?\b', normalized_text, re.I)
        if not m_age:
            m_age = re.search(r'([0-9]{1,3})\s*(?:years?|yrs?|yesrs)\b', normalized_text, re.I)
        if m_age:
            val = int(m_age.group(1))
            if 1 <= val <= 110:
                age = val
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting age: {e}")

    # Sex (Scans independently across document, wrapped in try/except for Bug K, robust regex for Bug L)
    sex = None
    try:
        if re.search(r'\b(?:mrs|ms|miss)\b', normalized_text, re.I):
            sex = "Female"
        elif re.search(r'\bmr\b', normalized_text, re.I):
            sex = "Male"
        else:
            m_combo = re.search(r'(?:age\s*[\/\\]\s*)?(?:sex|gender)\s*[:\-\|\=\.]*\s*(?:[0-9]{1,3}\s*(?:years?|yrs?|y)?\s*[\/\\]\s*)?(m[as]le|fe?male|[mf]\b)', normalized_text, re.I)
            if m_combo:
                raw_val = m_combo.group(1).lower()
                if raw_val in ['m', 'male', 'msle']:
                    sex = "Male"
                elif raw_val in ['f', 'female', 'femail']:
                    sex = "Female"
            if not sex:
                m_slash = re.search(r'\b[0-9]{1,3}\s*(?:years?|yrs?|y)?\s*[\/\\]\s*([mf]\b|male|female)', normalized_text, re.I)
                if m_slash:
                    raw_val = m_slash.group(1).lower()
                    if raw_val in ['m', 'male']:
                        sex = "Male"
                    elif raw_val in ['f', 'female']:
                        sex = "Female"
            if not sex:
                if re.search(r'\bfemale\b', normalized_text, re.I):
                    sex = "Female"
                elif re.search(r'\bmale\b', normalized_text, re.I):
                    sex = "Male"
    except Exception as e:
        logger.exception(f"[Step 1 Extraction] Failed extracting sex: {e}")

    # Confidence assessment
    has_primary = (fasting_glucose is not None) or (hba1c is not None) or (random_glucose is not None) or (bmi.get("value") is not None) or (height_cm is not None and weight_kg is not None)
    confidence = "high" if has_primary else "low"

    return {
        "fasting_glucose": fasting_glucose,
        "hba1c": hba1c,
        "random_glucose": random_glucose,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "waist_circumference_cm": waist_circumference_cm,
        "age": age,
        "sex": sex,
        "extraction_confidence": confidence,
        "raw_text_char_count": char_count
    }

# =====================================================================
# STEP 2: PRE-GENERATION GATE (CODE — Hard Stop, Not Optional)
# =====================================================================

def step2_metabolic_pre_generation_gate(
    raw_ocr_text: str,
    extracted_json: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Step 2 Hard Pre-generation Gate (Code only):
    1. If fasting_glucose, hba1c, random_glucose, AND bmi are ALL null AND cannot compute BMI -> STOP.
    2. If extraction_confidence == 'low' -> STOP.
    3. BMI CALCULATION: In code ONLY, if bmi.value is null but height_cm and weight_kg are non-null:
       compute BMI = weight_kg / (height_m)^2, tag 'source: calculated'.
    4. String/regex verify all extracted numbers against raw OCR text. Drop any that fail.
    5. Returns (gate_passed, error_message, verified_json).
    """
    normalized_ocr = unicodedata.normalize('NFKC', raw_ocr_text).replace('：', ':').replace('，', ',')
    fg = extracted_json.get("fasting_glucose")
    hba1c = extracted_json.get("hba1c")
    rg = extracted_json.get("random_glucose")
    bmi = extracted_json.get("bmi") or {"value": None, "source": "not_present"}
    ht = extracted_json.get("height_cm")
    wt = extracted_json.get("weight_kg")
    confidence = extracted_json.get("extraction_confidence", "low")

    # Check if BMI can be calculated in code
    can_compute_bmi = (bmi.get("value") is None) and (ht is not None) and (wt is not None) and (ht > 0)
    if can_compute_bmi:
        height_m = ht / 100.0
        computed_bmi = round(wt / (height_m * height_m), 1)
        flagged = (computed_bmi >= 23.0 or computed_bmi < 18.5)
        bmi = {
            "value": computed_bmi,
            "unit": "kg/m²",
            "reference_range": "18.5 - 22.9 kg/m² (WHO Asian cutoff)",
            "source_flagged": flagged,
            "source": "calculated"
        }
        logger.info(f"[Step 2 Code Gate] Computed BMI deterministically in code: {computed_bmi} kg/m² from ht {ht}cm, wt {wt}kg")

    # Hard Stop Check 1: Are all diabetes and obesity indicators null?
    has_any_indicator = (
        (fg is not None and fg.get("value") is not None) or
        (hba1c is not None and hba1c.get("value") is not None) or
        (rg is not None and rg.get("value") is not None) or
        (bmi is not None and bmi.get("value") is not None)
    )

    if not has_any_indicator or confidence == "low":
        err_msg = (
            "Could not extract diabetes or obesity-relevant values from this report. "
            "Please upload a report with glucose/HbA1c results or height & weight / BMI."
        )
        logger.error(f"[Step 2 Pre-generation Gate STOP] {err_msg}")
        return False, err_msg, extracted_json

    # String-match verification of extracted values against raw OCR text
    verified_json = {
        "fasting_glucose": None,
        "hba1c": None,
        "random_glucose": None,
        "height_cm": None,
        "weight_kg": None,
        "bmi": bmi,
        "waist_circumference_cm": None,
        "age": extracted_json.get("age"),
        "sex": extracted_json.get("sex"),
        "required_flagged_checklist": []
    }

    # Verify Fasting Glucose
    if fg and fg.get("value") is not None:
        val = fg["value"]
        orig = fg.get("original_value")
        v_str = str(int(val)) if val == int(val) else str(val)
        orig_str = str(orig) if orig is not None else None
        if re.search(rf'\b{re.escape(v_str)}\b', normalized_ocr) or (orig_str and (orig_str in normalized_ocr or re.search(rf'\b{re.escape(orig_str)}\b', normalized_ocr))):
            verified_json["fasting_glucose"] = fg
            if fg.get("source_flagged"):
                verified_json["required_flagged_checklist"].append("Fasting Blood Glucose")
        else:
            logger.warning(f"[Step 2 Verification] Fasting glucose {v_str} failed OCR string match, dropped.")

    # Verify HbA1c
    if hba1c and hba1c.get("value") is not None:
        v_str = str(hba1c["value"])
        if v_str in normalized_ocr or re.search(rf'\b{re.escape(v_str)}\b', normalized_ocr):
            verified_json["hba1c"] = hba1c
            if hba1c.get("source_flagged"):
                verified_json["required_flagged_checklist"].append("HbA1c")
        else:
            logger.warning(f"[Step 2 Verification] HbA1c {v_str} failed OCR string match, dropped.")

    # Verify Random Glucose
    if rg and rg.get("value") is not None:
        val = rg["value"]
        orig = rg.get("original_value")
        v_str = str(int(val)) if val == int(val) else str(val)
        orig_str = str(orig) if orig is not None else None
        if re.search(rf'\b{re.escape(v_str)}\b', normalized_ocr) or (orig_str and (orig_str in normalized_ocr or re.search(rf'\b{re.escape(orig_str)}\b', normalized_ocr))):
            verified_json["random_glucose"] = rg
            if rg.get("source_flagged"):
                verified_json["required_flagged_checklist"].append("Random Blood Glucose")
        else:
            logger.warning(f"[Step 2 Verification] Random glucose {v_str} failed OCR string match, dropped.")

    # Verify Height
    if ht is not None:
        ht_str = str(int(ht)) if ht == int(ht) else str(ht)
        ht_m_str = f"{ht / 100.0:.2f}"
        ht_m_str_1 = f"{ht / 100.0:.1f}"
        if re.search(rf'\b{re.escape(ht_str)}\b', normalized_ocr) or re.search(rf'\b{re.escape(ht_m_str)}\b', normalized_ocr) or re.search(rf'\b{re.escape(ht_m_str_1)}\b', normalized_ocr):
            verified_json["height_cm"] = ht
        else:
            logger.warning(f"[Step 2 Verification] Height {ht} not verified in OCR, dropped.")

    # Verify Weight
    if wt is not None:
        wt_str = str(int(wt)) if wt == int(wt) else str(wt)
        if re.search(rf'\b{re.escape(wt_str)}\b', normalized_ocr):
            verified_json["weight_kg"] = wt
        else:
            logger.warning(f"[Step 2 Verification] Weight {wt} not verified in OCR, dropped.")

    # Verify Printed BMI if source was "printed"
    if bmi and bmi.get("source") == "printed" and bmi.get("value") is not None:
        bmi_str = str(bmi["value"])
        if bmi_str in normalized_ocr or re.search(rf'\b{re.escape(bmi_str)}\b', normalized_ocr):
            verified_json["bmi"] = bmi
            if bmi.get("source_flagged"):
                verified_json["required_flagged_checklist"].append("BMI")
        else:
            logger.warning(f"[Step 2 Verification] Printed BMI {bmi_str} not verified in OCR, dropped.")
            verified_json["bmi"] = {"value": None, "source": "not_present"}
    elif bmi and bmi.get("source") == "calculated" and bmi.get("value") is not None:
        # Code-calculated BMI is valid
        verified_json["bmi"] = bmi
        if bmi.get("source_flagged"):
            verified_json["required_flagged_checklist"].append("BMI")

    # Verify Waist Circumference
    wc = extracted_json.get("waist_circumference_cm")
    if wc is not None:
        wc_str = str(int(wc)) if wc == int(wc) else str(wc)
        if re.search(rf'\b{re.escape(wc_str)}\b', normalized_ocr):
            verified_json["waist_circumference_cm"] = wc
            sex_val = (verified_json.get("sex") or "male").lower()
            cutoff = 90.0 if "m" in sex_val and "fe" not in sex_val else 80.0
            if wc > cutoff:
                verified_json["required_flagged_checklist"].append("Waist Circumference (Central Obesity)")

    # Final sanity check: do we have at least one verified indicator?
    has_verified = (
        (verified_json["fasting_glucose"] is not None) or
        (verified_json["hba1c"] is not None) or
        (verified_json["random_glucose"] is not None) or
        (verified_json["bmi"].get("value") is not None)
    )

    if not has_verified:
        err_msg = (
            "Could not extract diabetes or obesity-relevant values from this report. "
            "Please upload a report with glucose/HbA1c results or height & weight / BMI."
        )
        return False, err_msg, verified_json

    return True, None, verified_json

# =====================================================================
# STEP 3: RISK SCORING (ADA Criteria + WHO Asian-adjusted BMI Cutoffs)
# =====================================================================

def step3_metabolic_risk_scoring(verified_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3 Clinical Risk Scoring:
    - Conditioned ONLY on verified JSON.
    - DIABETES classification (ADA criteria):
      * Fasting Glucose >=126 mg/dL -> 'Diagnostic range for diabetes'
      * Fasting Glucose 100-125 -> 'Prediabetes range'
      * Fasting Glucose <100 -> 'Normal range'
      * HbA1c >=6.5% -> 'Diagnostic range for diabetes'
      * HbA1c 5.7-6.4% -> 'Prediabetes range'
      * HbA1c <5.7% -> 'Normal range'
      * If glucose and HbA1c disagree, report both separately — never average.
    - OBESITY classification (WHO Asian-adjusted cutoffs):
      * BMI <18.5 -> Underweight
      * BMI 18.5-22.9 -> Normal
      * BMI 23-24.9 -> Overweight (Asian cutoff; standard WHO cutoff is 25-29.9)
      * BMI >=25 -> Obese (Asian cutoff; standard WHO cutoff is >=30)
      * Waist >90cm (men) or >80cm (women) -> flag central obesity as additional modifier.
    - COMBINED risk note in plain language (no invented blended percentage).
    """
    user_prompt = f"Verified Step 2 Metabolic JSON:\n{json.dumps(verified_json, indent=2)}"

    llm_result = call_metabolic_llm(
        system_prompt=METABOLIC_STEP3_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )
    if llm_result and isinstance(llm_result, dict) and "diabetes_assessment" in llm_result:
        verified_fields_chk = [
            verified_json.get("fasting_glucose"),
            verified_json.get("hba1c"),
            verified_json.get("random_glucose"),
            verified_json.get("bmi", {}).get("value") if isinstance(verified_json.get("bmi"), dict) else verified_json.get("bmi"),
            verified_json.get("height_cm"),
            verified_json.get("weight_kg"),
            verified_json.get("waist_circumference_cm")
        ]
        cnt = sum(1 for x in verified_fields_chk if x is not None)
        comb_txt = str(llm_result.get("combined_risk_note", "")).lower()
        if cnt >= 2 and ("fewer than two" in comb_txt or "data-limited" in comb_txt or "data limited" in comb_txt):
            logger.warning("[Bug M Guard] Overriding LLM data-limited note with verified clinical synthesis.")
        else:
            return llm_result

    # High-precision deterministic clinical reasoning engine implementing Step 3 rules
    fg = verified_json.get("fasting_glucose")
    hba1c = verified_json.get("hba1c")
    rg = verified_json.get("random_glucose")
    bmi = verified_json.get("bmi") or {}
    wc = verified_json.get("waist_circumference_cm")
    sex = verified_json.get("sex") or "Male"
    is_male = "m" in sex.lower() and "fe" not in sex.lower()

    # Count total available indicators across both metabolic and anthropometric fields verified in Step 2 (Fix Bug C)
    verified_fields = [
        fg.get("value") if fg else None,
        hba1c.get("value") if hba1c else None,
        rg.get("value") if rg else None,
        bmi.get("value") if bmi else None,
        verified_json.get("height_cm"),
        verified_json.get("weight_kg"),
        verified_json.get("waist_circumference_cm")
    ]
    available_fields_count = sum(1 for x in verified_fields if x is not None)
    is_data_limited = available_fields_count < 2

    # 1. DIABETES CLASSIFICATION (ADA Criteria)
    diabetes_classifications = []
    diabetes_summary = ""
    glucose_status = None
    hba1c_status = None

    if fg and fg.get("value") is not None:
        val = fg["value"]
        if val >= 126.0:
            glucose_status = "Diagnostic range for diabetes"
        elif val >= 100.0:
            glucose_status = "Prediabetes range"
        else:
            glucose_status = "Normal range"
        diabetes_classifications.append({
            "marker": "Fasting Blood Glucose",
            "value": f"{val} mg/dL",
            "classification": glucose_status,
            "diagnostic_threshold": "ADA Standard: >=126 mg/dL Diabetes, 100-125 mg/dL Prediabetes",
            "source_flagged": fg.get("source_flagged", False)
        })

    if hba1c and hba1c.get("value") is not None:
        val = hba1c["value"]
        if val >= 6.5:
            hba1c_status = "Diagnostic range for diabetes"
        elif val >= 5.7:
            hba1c_status = "Prediabetes range"
        else:
            hba1c_status = "Normal range"
        diabetes_classifications.append({
            "marker": "HbA1c (Glycated Hemoglobin)",
            "value": f"{val}%",
            "classification": hba1c_status,
            "diagnostic_threshold": "ADA Standard: >=6.5% Diabetes, 5.7-6.4% Prediabetes",
            "source_flagged": hba1c.get("source_flagged", False)
        })

    if rg and rg.get("value") is not None and not fg:
        val = rg["value"]
        rg_status = "Diagnostic range for diabetes" if val >= 200.0 else ("Elevated Glycemia" if val >= 140.0 else "Normal range")
        diabetes_classifications.append({
            "marker": "Random Blood Glucose",
            "value": f"{val} mg/dL",
            "classification": rg_status,
            "diagnostic_threshold": "ADA Standard: >=200 mg/dL with symptoms",
            "source_flagged": rg.get("source_flagged", False)
        })

    # If both glucose and HbA1c are present, report both separately, never average
    if glucose_status and hba1c_status:
        if glucose_status == hba1c_status:
            diabetes_summary = f"Both fasting glucose and HbA1c concordantly indicate {glucose_status}."
        else:
            diabetes_summary = f"Fasting glucose falls in the {glucose_status}, while HbA1c indicates {hba1c_status} (reported separately per ADA clinical guidelines)."
    elif glucose_status:
        diabetes_summary = f"Fasting glucose indicates {glucose_status}."
    elif hba1c_status:
        diabetes_summary = f"HbA1c indicates {hba1c_status}."
    else:
        diabetes_summary = "Cannot be assessed (glycemic panel pending)."

    # 2. OBESITY CLASSIFICATION (WHO Asian-adjusted cutoffs)
    obesity_assessment = {}
    bmi_val = bmi.get("value") if bmi else None

    if bmi_val is not None:
        if bmi_val < 18.5:
            obesity_cat = "Underweight"
        elif bmi_val <= 22.9:
            obesity_cat = "Normal"
        elif bmi_val <= 24.9:
            obesity_cat = "Overweight (Asian cutoff; standard WHO cutoff is 25–29.9)"
        else:
            obesity_cat = "Obese (Asian cutoff; standard WHO cutoff is >=30)"

        central_obesity = False
        central_obesity_note = None
        if wc is not None:
            cutoff = 90.0 if is_male else 80.0
            if wc > cutoff:
                central_obesity = True
                central_obesity_note = (
                    f"Waist circumference of {wc} cm exceeds the {int(cutoff)} cm Asian cutoff for "
                    f"{'men' if is_male else 'women'}, flagging central adiposity as an independent metabolic risk modifier."
                )

        obesity_assessment = {
            "bmi_value": bmi_val,
            "bmi_unit": "kg/m²",
            "classification": obesity_cat,
            "standard_used": "WHO / ICMR-NIN Asian-Adjusted BMI Cutoffs",
            "bmi_source": bmi.get("source", "printed"),
            "central_obesity": central_obesity,
            "central_obesity_note": central_obesity_note,
            "source_flagged": bmi.get("source_flagged", False)
        }

    # 3. COMBINED RISK NOTE (Plain language, NO invented single percentage)
    # Check if all available markers are in the Normal range (Fix Bug F)
    is_glycemic_normal = (
        (glucose_status is None or "Normal" in glucose_status) and
        (hba1c_status is None or "Normal" in hba1c_status)
    )
    is_obesity_normal = (
        obesity_assessment.get("classification") == "Normal" and
        not obesity_assessment.get("central_obesity", False)
    )
    is_all_normal = is_glycemic_normal and is_obesity_normal and (glucose_status is not None or hba1c_status is not None) and (bmi_val is not None)

    if is_data_limited:
        combined_risk_note = (
            "Assessment is data-limited as only a single metabolic or anthropometric field was available in this report. "
            "Additional laboratory or anthropometric measurements are recommended before drawing confident clinical conclusions."
        )
    elif is_all_normal:
        combined_risk_note = (
            f"All monitored metabolic biomarkers (fasting glucose and BMI {bmi_val} kg/m²) are within healthy normal reference ranges. "
            "Metabolic health profile is optimal; continue maintaining balanced nutrition and regular physical activity to sustain baseline health."
        )
    elif bmi_val is not None and (glucose_status or hba1c_status):
        bmi_short_cat = obesity_assessment.get("classification", "").split("(")[0].strip()
        if glucose_status and hba1c_status and glucose_status != hba1c_status:
            combined_risk_note = (
                f"Adiposity evaluation ({bmi_short_cat}: BMI {bmi_val} kg/m²) is accompanied by discordant glycemic markers: "
                f"fasting glucose falls in the {glucose_status}, while HbA1c indicates {hba1c_status} (reported separately per ADA clinical guidelines; repeat confirmatory testing recommended)."
            )
        elif "Normal" in (glucose_status or "") and "Normal" in (hba1c_status or "") and bmi_short_cat == "Normal":
            combined_risk_note = (
                f"All monitored biomarkers are within standard reference ranges (BMI {bmi_val} kg/m²: Normal; Glycemia: {glucose_status or hba1c_status}). "
                "Metabolic parameters show healthy baseline regulation."
            )
        elif "Diagnostic" in (glucose_status or "") or "Diagnostic" in (hba1c_status or "") or "Obese" in bmi_short_cat:
            glycemic_desc = glucose_status or hba1c_status
            combined_risk_note = (
                f"Adiposity evaluation ({bmi_short_cat}: BMI {bmi_val} kg/m²) and glycemic assessment ({glycemic_desc}) "
                f"concordantly indicate elevated metabolic risk requiring targeted nutritional and lifestyle guidance."
            )
        else:
            glycemic_desc = glucose_status or hba1c_status
            combined_risk_note = (
                f"Adiposity evaluation ({bmi_short_cat}: BMI {bmi_val} kg/m²) and glycemic assessment ({glycemic_desc}) "
                f"indicate borderline or moderate metabolic risk. Preventive diet and routine activity can restore optimal glycemic balance."
            )
    elif bmi_val is not None:
        bmi_short_cat = obesity_assessment.get("classification", "").split("(")[0].strip()
        central_clause = f" with central adiposity (waist {wc} cm)" if obesity_assessment.get("central_obesity") else ""
        if bmi_short_cat == "Normal":
            combined_risk_note = (
                f"Body Mass Index ({bmi_val} kg/m²) is within the healthy normal range per WHO Asian guidelines. "
                "Laboratory glycemic metrics (fasting glucose / HbA1c) were not included in this panel."
            )
        else:
            combined_risk_note = (
                f"Adiposity assessment indicates {obesity_assessment.get('classification')}{central_clause}. "
                f"Glycemic metrics (fasting glucose / HbA1c) are pending to evaluate complete metabolic risk."
            )
    elif glucose_status or hba1c_status:
        glycemic_desc = glucose_status or hba1c_status
        if "Normal" in glycemic_desc:
            combined_risk_note = (
                f"Glycemic assessment indicates {diabetes_summary} "
                "Anthropometric measurements (BMI / waist circumference) were not included in this report."
            )
        else:
            combined_risk_note = (
                f"Glycemic assessment indicates {diabetes_summary} "
                f"Anthropometric measurements (BMI / waist circumference) are recommended to evaluate complete metabolic risk."
            )
    else:
        combined_risk_note = "Insufficient data to establish combined metabolic risk note."

    # Automated Consistency Assertion (Fix Bug F): If all classifications are normal, guarantee no risk-escalating language
    if is_all_normal or (is_glycemic_normal and is_obesity_normal):
        forbidden_risk_terms = ["compounds", "elevated concern", "requiring intervention", "insulin resistance", "elevated metabolic concern"]
        for term in forbidden_risk_terms:
            if term in combined_risk_note.lower():
                combined_risk_note = (
                    f"All monitored metabolic biomarkers (fasting glucose and BMI {bmi_val} kg/m²) are within healthy normal reference ranges. "
                    "Metabolic health profile is optimal; continue maintaining balanced nutrition and regular physical activity to sustain baseline health."
                )
                break

    return {
        "module": "metabolic_risk_diabetes_obesity",
        "data_limited": is_data_limited,
        "diabetes_assessment": {
            "summary": diabetes_summary,
            "classifications": diabetes_classifications,
            "fasting_glucose_status": glucose_status,
            "hba1c_status": hba1c_status
        },
        "obesity_assessment": obesity_assessment,
        "combined_risk_note": combined_risk_note,
        "source_flagged_confirmed": [
            item["marker"] for item in diabetes_classifications if item.get("source_flagged")
        ] + ([f"BMI ({bmi_val} kg/m²)"] if obesity_assessment.get("source_flagged") else [])
    }

# =====================================================================
# STEP 4: POST-GENERATION AUDIT GATE (CODE)
# =====================================================================

def step4_metabolic_post_generation_audit(
    verified_json: Dict[str, Any],
    step3_output: Dict[str, Any],
    filename: str = "report"
) -> Dict[str, Any]:
    """
    Step 4 Programmatic Post-generation Audit Gate:
    1. Regex-extract every number in Step 3 output; confirm each exists in Step 2 JSON
       (including code-calculated BMI).
    2. Confirm every source_flagged field appears in output text.
    3. If either check fails: block rendering, log failure with filename + both JSONs.
    4. Only display "Verified"/"Audit Passed" if this check actually ran.
    """
    audit_passed = True
    audit_issues = []

    # 1. Ground truth numbers from Step 2 JSON
    verified_numbers = set()
    fg = verified_json.get("fasting_glucose")
    if fg and fg.get("value") is not None:
        verified_numbers.add(float(fg["value"]))
        verified_numbers.add(float(int(fg["value"])))
        if fg.get("original_value") is not None:
            verified_numbers.add(float(fg["original_value"]))

    hba1c = verified_json.get("hba1c")
    if hba1c and hba1c.get("value") is not None:
        verified_numbers.add(float(hba1c["value"]))

    rg = verified_json.get("random_glucose")
    if rg and rg.get("value") is not None:
        verified_numbers.add(float(rg["value"]))
        verified_numbers.add(float(int(rg["value"])))
        if rg.get("original_value") is not None:
            verified_numbers.add(float(rg["original_value"]))

    bmi = verified_json.get("bmi")
    if bmi and bmi.get("value") is not None:
        verified_numbers.add(float(bmi["value"]))

    if verified_json.get("height_cm") is not None:
        verified_numbers.add(float(verified_json["height_cm"]))

    if verified_json.get("weight_kg") is not None:
        verified_numbers.add(float(verified_json["weight_kg"]))

    if verified_json.get("waist_circumference_cm") is not None:
        verified_numbers.add(float(verified_json["waist_circumference_cm"]))

    if verified_json.get("age") is not None:
        verified_numbers.add(float(verified_json["age"]))

    # Add standard clinical threshold numbers that the prompt dictates using
    standard_cutoffs = {
        70.0, 99.0, 100.0, 125.0, 126.0, 139.0, 140.0, 200.0,
        3.5, 3.9, 5.5, 5.6, 5.7, 6.4, 6.5, 7.8, 18.0, 18.0182,
        18.5, 22.9, 23.0, 24.9, 25.0, 29.9, 30.0,
        80.0, 90.0, 1.0, 2.0
    }

    # Extract all numbers from Step 3 text and structured payload
    step3_full_text = (
        step3_output.get("combined_risk_note", "") + " " +
        step3_output.get("diabetes_assessment", {}).get("summary", "") + " " +
        " ".join([
            f"{c.get('marker', '')} {c.get('value', '')} {c.get('diagnostic_threshold', '')}"
            for c in step3_output.get("diabetes_assessment", {}).get("classifications", [])
        ]) + " " +
        str(step3_output.get("obesity_assessment", {}))
    )

    found_numbers = re.findall(r'\b[0-9]+(?:\.[0-9]+)?\b', step3_full_text)

    for num_str in found_numbers:
        val = float(num_str)
        if val in verified_numbers or val in standard_cutoffs:
            continue
        audit_passed = False
        issue = f"Ungrounded numeric claim '{num_str}' in Step 3 output not found in verified Step 2 JSON."
        audit_issues.append(issue)
        logger.error(f"[Step 4 Post-Generation Audit FAIL] {issue}")

    # 2. Confirm every source_flagged field appears in Step 3 output text
    text_lower = step3_full_text.lower()
    checklist = verified_json.get("required_flagged_checklist", [])

    for item in checklist:
        key = item.lower().split("(")[0].strip()
        if "glucose" in key and "glucose" in text_lower:
            continue
        if "hba1c" in key and "hba1c" in text_lower:
            continue
        if "bmi" in key and ("bmi" in text_lower or "obese" in text_lower or "overweight" in text_lower):
            continue
        if "waist" in key and ("waist" in text_lower or "central" in text_lower):
            continue
        audit_passed = False
        issue = f"Source-flagged abnormality '{item}' omitted from Step 3 clinical output."
        audit_issues.append(issue)
        logger.error(f"[Step 4 Post-Generation Audit FAIL] {issue}")

    if not audit_passed:
        logger.error(
            f"[Step 4 Blocked Report] Filename: {filename}\n"
            f"Step 2 JSON: {json.dumps(verified_json)}\n"
            f"Step 3 Output: {json.dumps(step3_output)}\n"
            f"Audit Issues: {audit_issues}"
        )

    return {
        "audit_passed": audit_passed,
        "audit_issues": audit_issues,
        "checklist_verified": checklist,
        "filename": filename,
        "audit_label": "Verified / Audit Passed" if audit_passed else "Audit Discrepancy Blocked"
    }

# =====================================================================
# FULL 4-STEP METABOLIC PIPELINE ORCHESTRATOR
# =====================================================================

def execute_metabolic_pipeline(raw_ocr_text: str, filename: str) -> Dict[str, Any]:
    """
    Executes the complete 4-step metabolic risk pipeline:
      STEP 1: Extraction (strict, no derivations)
      STEP 2: Pre-generation gate (code hard-stop, code-only BMI calculation)
      STEP 3: Risk scoring (ADA + WHO Asian cutoffs)
      STEP 4: Post-generation audit (code number & checklist validation)
    """
    logger.info(f"Executing metabolic pipeline for {filename}")

    # STEP 1: Extraction
    step1_res = step1_metabolic_extraction(raw_ocr_text)

    # STEP 2: Pre-generation Gate
    gate_passed, err_msg, verified_json = step2_metabolic_pre_generation_gate(raw_ocr_text, step1_res)
    if not gate_passed:
        return {
            "success": False,
            "error": err_msg,
            "pipeline_status": "HARD_STOP_PRE_GENERATION",
            "filename": filename,
            "step1_extraction": step1_res,
            "step4_audit": {
                "audit_passed": False,
                "audit_label": "Pre-Generation Gate Stopped"
            }
        }

    # STEP 3: Risk Scoring & Classification
    step3_res = step3_metabolic_risk_scoring(verified_json)

    # STEP 4: Post-generation Audit Gate
    audit_res = step4_metabolic_post_generation_audit(verified_json, step3_res, filename)

    return {
        "success": audit_res["audit_passed"],
        "pipeline_version": "metabolic_v1_ada_who",
        "pipeline_status": "APPROVED" if audit_res["audit_passed"] else "BLOCKED_FOR_REVIEW",
        "filename": filename,
        "step1_extraction": step1_res,
        "step2_verified_json": verified_json,
        "step3_risk_scoring": step3_res,
        "step4_audit": audit_res,
        # Direct convenience views for frontend UI
        "data_limited": step3_res.get("data_limited", False),
        "diabetes_assessment": step3_res["diabetes_assessment"],
        "obesity_assessment": step3_res["obesity_assessment"],
        "combined_risk_note": step3_res["combined_risk_note"],
        "audit_passed": audit_res["audit_passed"],
        "audit_label": audit_res["audit_label"],
        "verified_vitals": {
            "fasting_glucose": verified_json.get("fasting_glucose"),
            "hba1c": verified_json.get("hba1c"),
            "random_glucose": verified_json.get("random_glucose"),
            "bmi": verified_json.get("bmi"),
            "height_cm": verified_json.get("height_cm"),
            "weight_kg": verified_json.get("weight_kg"),
            "waist_circumference_cm": verified_json.get("waist_circumference_cm"),
            "age": verified_json.get("age"),
            "sex": verified_json.get("sex")
        }
    }
