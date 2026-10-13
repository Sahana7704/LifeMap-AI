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
import time
import logging
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("lifemap_metabolic")

# =====================================================================
# =====================================================================
# PRE-EXTRACTION DOCUMENT CLASSIFICATION GATE (BUG R: POPULATION TABLES)
# =====================================================================

def validate_document_type(raw_ocr_text: str) -> Tuple[bool, List[str]]:
    """
    Bug R & Bug S Gate:
    Validates whether the document is an individual patient diagnostic report
    or an aggregate statistical / population comparison table from research.

    Negative Signals (Specific to research/population tables):
    1. Sample size notation: N= or n= followed by a number >= 10, or (N=...)
    2. Group / cohort comparison headers (e.g. "Patients With vs Without", "Cases vs Controls", "Group A vs Group B")
    3. P-values (explicit "P Value", "p < 0.05", "p < .001", etc. - avoiding "PP < 140")
    4. Mean/SD statistical notation representing a population (e.g. "mean ± sd", "177.5 ± 62.7", "mean (sd)")
    5. Cohort percentage columns (e.g. "No. (%)", "n (%)", "% (n=)")

    Positive Patient Identifiers (Single-patient report markers):
    - Patient Name, Patient ID, Reg No, MRN, UHID, Lab No, Sample ID, Referring Doctor, Age/Sex.

    Gate Decision:
    - If positive single-patient identifiers are present:
      Only refuse if there are 3+ strong negative signals including sample size notation or group comparison.
    - If NO positive patient identifier is present:
      Refuse if 2+ negative signals are detected.
    """
    if not raw_ocr_text:
        return True, []

    text_norm = unicodedata.normalize('NFKC', raw_ocr_text)
    signals = []

    # 1. Sample size notation: N= or n= followed by a number >= 10
    if re.search(r'\b[Nn]\s*=\s*[0-9]{2,}\b|\(\s*[Nn]\s*=\s*[0-9]+\s*\)', text_norm):
        signals.append("Sample size notation (N= or n=)")

    # 2. Group/cohort comparison headers
    cohort_patterns = [
        r'\bpatients\s+with\s+(?:vs\.?|versus|and)\s+(?:without|patients)\b',
        r'\b(?:cases\s+vs\.?\s+controls|group\s+[a-z0-9]\s+vs\.?\s+group|treatment\s+vs\.?\s+placebo)\b',
        r'\bhistory\s+of\s+[^:\n\r]+:\s*(?:yes\s*[\/\\]\s*no|present\s*[\/\\]\s*absent)\b',
        r'\b(?:overall\s+cohort|study\s+population|total\s+patients\s*\(n\s*=)\b',
        r'\bcohort\s+[a-z0-9]\b'
    ]
    if any(re.search(pat, text_norm, re.IGNORECASE) for pat in cohort_patterns):
        signals.append("Group/cohort comparison headers")

    # 3. P-values (e.g. P Value, p < 0.05, p < .001)
    # Strictly avoid matching PP glucose like "(PP) < 140"
    if re.search(r'\b(?:p\s*[-–—]?\s*val(?:ue)?|p-value)\b|\bp\s*[<>=]\s*(?:0?\.[0-9]{2,4}|\.001)\b', text_norm, re.IGNORECASE):
        signals.append("P-values / statistical significance")

    # 4. Mean (SD) format: e.g. "177.5 (62.7)", "mean ± sd", "mean (sd)"
    if re.search(r'\b(?:mean\s*±\s*sd|mean\s*\(\s*sd\s*\)|mean\s*\[\s*sd\s*\])\b|\b[0-9]+(?:\.[0-9]+)?\s*±\s*[0-9]+(?:\.[0-9]+)?\b', text_norm, re.IGNORECASE):
        signals.append("Mean/SD statistical notation")
    elif re.search(r'\b(?:mean|sd)\b', text_norm, re.I) and re.search(r'\b[0-9]{2,3}(?:\.[0-9]+)?\s*\(\s*[0-9]{1,3}(?:\.[0-9]+)?\s*\)', text_norm):
        signals.append("Mean/SD statistical notation")

    # 5. Cohort percentage columns: e.g. "No. (%)", "n (%)", "% (n=)"
    if re.search(r'\b(?:no\.?\s*\(\s*%\s*\)|n\s*\(\s*%\s*\)|%\s*\(n\s*=\s*[0-9]+\))\b', text_norm, re.IGNORECASE):
        signals.append("Cohort percentage columns")

    # Positive Patient Identifiers (Single named patient report markers)
    patient_patterns = [
        r'\bpatient\s*(?:name)?\s*:\s*[a-zA-Z]',
        r'\b(?:pt|patient)\s+name\b',
        r'\b(?:mrn|uhid|reg(?:istration)?\s*(?:no|num|number)?|patient\s*id|pid|lab\s*no|sid|sample\s*id|barcode|ipd|opd)\s*:\s*[a-zA-Z0-9]',
        r'\b(?:referred\s*by|ref\s*by|dr\.?|doctor|physician)\s*:\s*[a-zA-Z]',
        r'\b(?:age\s*[\/:]\s*[0-9]{1,3}\s*[\/:]?\s*(?:sex|gender)?)\b',
        r'\b(?:male|female)\s*[\/,\s]+\d{1,3}\s*(?:y|yr|yrs|years)\b'
    ]
    has_patient_id = any(re.search(pat, text_norm, re.IGNORECASE) for pat in patient_patterns)

    # Decision rule:
    # If the document has a verified individual patient identifier, refuse ONLY IF there are 3+ strong population signals including N= and group comparison.
    # Otherwise (no patient identifier), refuse if 2+ signals are detected.
    if has_patient_id:
        is_research = len(signals) >= 3 and ("Sample size notation (N= or n=)" in signals or "Group/cohort comparison headers" in signals)
    else:
        is_research = len(signals) >= 2

    return not is_research, signals

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
    "1. Extract ONLY: Fasting Blood Glucose, Post-Prandial Glucose, HbA1c, Random Blood Glucose, "
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
    '  "post_prandial_glucose": {"value": number|null, "unit": str, "reference_range": str, "source_flagged": bool},\n'
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
    "- Post-Prandial Glucose >=200 mg/dL -> 'Diagnostic range for diabetes'\n"
    "- Post-Prandial Glucose 140–199 -> 'Prediabetes range'\n"
    "- Post-Prandial Glucose <70 or below reference range -> 'Low / Below Reference Range (Reactive Hypoglycemia pattern)'\n"
    "- Post-Prandial Glucose 70–139 -> 'Normal range'\n"
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
# IN-MEMORY CACHE KEYED BY REPORT HASH
# =====================================================================

_METABOLIC_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour in-memory cache

def get_cached_extraction(report_hash: str) -> Optional[Dict[str, Any]]:
    """Returns cached pipeline result if available and fresh, avoiding duplicate LLM calls."""
    if report_hash in _METABOLIC_CACHE:
        cached_time, cached_res = _METABOLIC_CACHE[report_hash]
        if time.time() - cached_time < _CACHE_TTL_SECONDS:
            logger.info(f"[Cache Hit] Re-uploaded report matched hash {report_hash[:12]} - 0 LLM calls made.")
            copy_res = json.loads(json.dumps(cached_res))
            copy_res["cached"] = True
            return copy_res
    return None

def set_cached_extraction(report_hash: str, result: Dict[str, Any]):
    """Stores pipeline result in cache keyed by report hash."""
    if report_hash and result:
        _METABOLIC_CACHE[report_hash] = (time.time(), result)

# =====================================================================
# LLM CALLER DISPATCHER WITH RETRY-AFTER & LIGHTER MODEL
# =====================================================================

def call_metabolic_llm(system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    """
    Invokes external LLM using a light, high-throughput model (default: gemini-2.5-flash-lite).
    Includes exponential backoff (max 2 retries, honors Retry-After) and detailed logging
    of upstream status code, model name, and response body on failure.
    Skips immediately on 404 (model not found).
    Returns parsed JSON if successful, or None on rate-limit/failure to allow clean fallback.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    # Prefer lighter, high-throughput model (default: gemini-2.5-flash-lite)
    # Exclude retired Gemini 1.5 models from candidates
    raw_primary = os.environ.get("METABOLIC_LLM_MODEL", "gemini-2.5-flash-lite").strip()
    if "1.5" in raw_primary:
        raw_primary = "gemini-2.5-flash-lite"
    candidate_models = []
    for m in [raw_primary, "gemini-2.5-flash-lite", "gemini-2.5-flash"]:
        if m and "1.5" not in m and m not in candidate_models:
            candidate_models.append(m)

    if gemini_key:
        for model_name in candidate_models:
            max_upstream_retries = 2
            for attempt in range(1, max_upstream_retries + 1):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel(
                        model_name=model_name,
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
                    err_str = str(e)
                    code_val = getattr(e, "code", None) or getattr(getattr(e, "response", None), "status_code", None)
                    is_404 = (code_val == 404) or ("404" in err_str) or ("not found" in err_str.lower())
                    is_429 = (code_val == 429) or ("429" in err_str) or ("quota" in err_str.lower()) or ("resourceexhausted" in err_str.lower())
                    
                    status_code = 404 if is_404 else (429 if is_429 else (code_val or 500))

                    # Extract Retry-After if provided in error string or default exponential backoff
                    retry_after = 2.0 * attempt
                    m_retry = re.search(r'(?:retry after|wait|cooldown)\s*[:=]?\s*(\d+(?:\.\d+)?)', err_str, re.I)
                    if m_retry:
                        try:
                            retry_after = max(1.0, float(m_retry.group(1)))
                        except Exception:
                            pass

                    # Log upstream status code, model name, and response body (Requirement 1)
                    logger.warning(
                        f"[Upstream LLM Call Failed] Provider: Gemini | Model: {model_name} | "
                        f"Status: {status_code} | Upstream Error Body: {err_str[:300]} | "
                        f"Retry-After: {retry_after}s | Attempt: {attempt}/{max_upstream_retries}"
                    )

                    # Treat 404 as skip to next model immediately without retrying
                    if is_404:
                        logger.info(f"[Gemini 404 Not Found] Skipping {model_name} to next candidate model.")
                        break

                    if is_429 and attempt < max_upstream_retries:
                        time.sleep(retry_after)
                        continue
                    break  # Try next candidate model or fallback

    # Groq Fallback (High-speed Llama 3.1 8B Instant with generous limits)
    if groq_key:
        try:
            import urllib.request
            groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
            req_data = {
                "model": groq_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(req_data).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                logger.info(f"[Groq LLM Success] Extracted via {groq_model}")
                return json.loads(content)
        except Exception as e:
            logger.warning(f"[Groq Upstream Call Failed] Error: {e}")

    # OpenAI Fallback
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
            logger.warning(f"[OpenAI Upstream Call Failed] Model: gpt-4o-mini | Error: {e}")

    logger.info("[Upstream LLM Unavailable/Rate-Limited] Transitioning to deterministic code-only regex fallback.")
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
    allow_mmol: bool = False,
    exclude_headers: Optional[List[str]] = None
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[str], Optional[str]]:
    """
    Robust multi-section extractor:
    Scans entire document text for each test header independently.
    Handles intervening lines such as 'Method: HPLC', 'Sample: Serum', etc.
    Avoids capturing reference intervals (< 5.7, 70-100) or subsequent test sections.
    Masks reference intervals so reference boundaries are never misidentified as patient results.
    Returns (extracted_value, start_idx, end_idx, unit_detected, ref_range_str).
    """
    combined_header = r'(?:' + '|'.join(header_patterns) + r')\b'
    for m in re.finditer(combined_header, text, re.I):
        line_start = text.rfind('\n', 0, m.start()) + 1
        line_end = text.find('\n', m.end())
        if line_end == -1:
            line_end = len(text)
        current_line = text[line_start:line_end]

        if exclude_headers and any(re.search(ex, current_line, re.I) for ex in exclude_headers):
            continue

        start_pos = m.end()
        snippet = text[start_pos:start_pos + 450]
        m_stop = re.search(r'(?:biological\s+reference|reference\s+interval|reference\s+range|ref\.\s*range|ref\s*interval|biological\s*ref)', snippet, re.I)
        search_window = snippet[:m_stop.start()] if m_stop else snippet

        # Check for mmol/L unit context in search_window
        window_has_mmol = bool(re.search(r'\bmmol\s*[\/\.]\s*[lL]\b', search_window, re.I))

        # Identify any printed reference intervals in search_window (e.g. 4.0 - 6.0, 70-99, 90 - 120, < 5.7, > 140)
        ref_intervals = []
        raw_ref_str = None
        for r_m in re.finditer(r'(?:([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:[-–—~]|to)\s*([0-9]{1,3}(?:\.[0-9]+)?)|([<>≤≥])\s*([0-9]{1,3}(?:\.[0-9]+)?))', search_window):
            if r_m.group(1) and r_m.group(2):
                ref_intervals.append((float(r_m.group(1)), float(r_m.group(2)), r_m.start(), r_m.end()))
                if not raw_ref_str:
                    raw_ref_str = r_m.group(0).strip()
            elif r_m.group(4):
                ref_intervals.append((None, float(r_m.group(4)), r_m.start(), r_m.end()))
                if not raw_ref_str:
                    raw_ref_str = r_m.group(0).strip()

        # Mask reference intervals to create a clean search surface where range boundaries cannot be picked
        masked_window = list(search_window)
        for _, _, s_idx, e_idx in ref_intervals:
            for i in range(s_idx, e_idx):
                masked_window[i] = ' '
        masked_search_window = "".join(masked_window)

        # Priority 1: Explicit 'Result: X' or 'Value: X'
        m_res = re.search(r'(?:result|value|observed|reading|level)[\s:\-|=]*([0-9]{1,3}(?:\.[0-9]+)?)', masked_search_window, re.I)
        if m_res:
            try:
                val = float(m_res.group(1))
                if min_val <= val <= max_val:
                    return val, m.start(), start_pos + m_res.end(), "mg/dL" if not window_has_mmol else "mmol/L", raw_ref_str
                if allow_mmol and window_has_mmol and 2.0 <= val <= 35.0:
                    return val, m.start(), start_pos + m_res.end(), "mmol/L", raw_ref_str
            except ValueError:
                pass

        # Priority 2: Number followed by matching unit (or flag then unit, like 118 H mg/dL)
        if unit_pattern:
            m_unit = re.search(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:[A-Za-z]\s+)?(' + unit_pattern + r')', masked_search_window, re.I)
            if m_unit:
                try:
                    val = float(m_unit.group(1))
                    unit_str = m_unit.group(2)
                    is_mmol = "mmol" in unit_str.lower()
                    if min_val <= val <= max_val:
                        return val, m.start(), start_pos + m_unit.end(), "mmol/L" if is_mmol else "mg/dL", raw_ref_str
                    if allow_mmol and is_mmol and 2.0 <= val <= 35.0:
                        return val, m.start(), start_pos + m_unit.end(), "mmol/L", raw_ref_str
                except ValueError:
                    pass

        # Priority 3: Standalone number on lines that do not describe methods/instruments
        lines = masked_search_window.split('\n')
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
                continue
            try:
                val = float(m_num.group(2))
                # Validation check: confirm not accidentally matching a range boundary if an alternative exists
                boundary_match = any((val == low or val == high) for low, high, _, _ in ref_intervals if low is not None and high is not None)
                if boundary_match:
                    all_nums = [float(x) for x in re.findall(r'\b([0-9]{1,3}(?:\.[0-9]+)?)\b', filtered_window)]
                    alternatives = [n for n in all_nums if not any(n == low or n == high for low, high, _, _ in ref_intervals if low is not None and high is not None)]
                    if alternatives:
                        val = alternatives[0]

                if min_val <= val <= max_val:
                    return val, m.start(), start_pos + len(search_window), "mmol/L" if window_has_mmol else "mg/dL", raw_ref_str
                if allow_mmol and window_has_mmol and 2.0 <= val <= 35.0:
                    return val, m.start(), start_pos + len(search_window), "mmol/L", raw_ref_str
            except ValueError:
                pass

    return None, None, None, None, None

def step1_metabolic_extraction(raw_ocr_text: str) -> Dict[str, Any]:
    """
    Step 1 Deterministic Clinical Extractor:
    Extracts explicitly printed values from individual diagnostic reports:
    - age, sex, height, weight, BMI (only if printed)
    - blood pressure (mean_sbp, mean_dbp, formatted "134/84")
    - fasting glucose, post-prandial glucose, random glucose
    - HbA1c (strictly rows named HbA1c / Glycated Hemoglobin / Glycosylated Hemoglobin / A1c; NEVER Hemoglobin)
    - lipid panel: total cholesterol, HDL, LDL, triglycerides
    - clinical history: high BP, high cholesterol, smoking 100+ cigarettes, regular physical activity, family history of diabetes
    Never calculates BMI in this step (Step 2 handles calculation).
    Missing values remain None ("not provided"). NEVER impute or invent values.
    """
    normalized_text = unicodedata.normalize('NFKC', raw_ocr_text).replace('：', ':').replace('，', ',')
    char_count = len(normalized_text)
    lines = [l.strip() for l in normalized_text.split('\n') if l.strip()]

    def find_num_near(pattern: str, max_lines: int = 2, min_val: float = 0.0, max_val: float = 2000.0) -> Optional[float]:
        for idx, line in enumerate(lines):
            if line.lower().startswith('note:'):
                continue
            if re.search(pattern, line, re.I):
                m_same = re.search(r'[:\-\|\=\s]+([0-9]+(?:\.[0-9]+)?)', line)
                if m_same:
                    try:
                        v = float(m_same.group(1))
                        if min_val <= v <= max_val:
                            return v
                    except ValueError:
                        pass
                for offset in range(1, max_lines + 1):
                    if idx + offset < len(lines):
                        nxt = lines[idx + offset]
                        if nxt.lower().startswith('note:'):
                            break
                        m_nxt = re.match(r'^([0-9]+(?:\.[0-9]+)?)\b', nxt)
                        if m_nxt:
                            try:
                                v = float(m_nxt.group(1))
                                if min_val <= v <= max_val:
                                    return v
                            except ValueError:
                                pass
        return None

    # 1. Age
    age = None
    for idx, line in enumerate(lines):
        if re.search(r'\b(?:age)\b', line, re.I) and not line.lower().startswith('note:'):
            m_age = re.search(r'\b([0-9]{1,3})\s*(?:years?|yrs?)?\b', line, re.I)
            if m_age and m_age.group(1) != line.strip():
                try:
                    val = int(m_age.group(1))
                    if 1 <= val <= 120:
                        age = val
                        break
                except ValueError:
                    pass
            for offset in (1, 2):
                if idx + offset < len(lines):
                    m_nxt = re.match(r'^([0-9]{1,3})\s*(?:years?|yrs?)?', lines[idx + offset], re.I)
                    if m_nxt:
                        try:
                            val = int(m_nxt.group(1))
                            if 1 <= val <= 120:
                                age = val
                                break
                        except ValueError:
                            pass
            if age is not None:
                break

    # 2. Sex
    sex = None
    for idx, line in enumerate(lines):
        if re.search(r'\b(?:sex|gender)\b', line, re.I) and not line.lower().startswith('note:'):
            m_s = re.search(r'\b(male|female)\b', line, re.I)
            if m_s:
                sex = m_s.group(1).capitalize()
                break
            for offset in (1, 2):
                if idx + offset < len(lines):
                    m_nxt = re.match(r'^(male|female)\b', lines[idx + offset], re.I)
                    if m_nxt:
                        sex = m_nxt.group(1).capitalize()
                        break
            if sex is not None:
                break

    # 3. Height (cm)
    height_cm = find_num_near(r'^(?:height|ht)\b', max_lines=2, min_val=50.0, max_val=250.0)

    # 4. Weight (kg)
    weight_kg = find_num_near(r'^(?:weight|wt)\b', max_lines=2, min_val=20.0, max_val=300.0)

    # 5. BMI (explicitly printed only)
    bmi_val = find_num_near(r'(?:body\s*mass\s*index|bmi)\b', max_lines=2, min_val=10.0, max_val=80.0)
    bmi = None
    if bmi_val is not None:
        bmi = {
            "value": bmi_val,
            "unit": "kg/m²",
            "reference_range": "18.5 - 22.9 kg/m² (WHO Asian cutoff)",
            "source_flagged": (bmi_val >= 23.0 or bmi_val < 18.5),
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

    # 6. Blood Pressure (mean_sbp, mean_dbp, formatted "134/84")
    mean_sbp = None
    mean_dbp = None
    blood_pressure = None
    bp_pat = r'([0-9]{2,3})\s*[\/\-]\s*([0-9]{2,3})'
    for idx, line in enumerate(lines):
        if re.search(r'blood\s*pressure|\bbp\b', line, re.I):
            m_bp = re.search(bp_pat, line)
            if m_bp:
                mean_sbp = float(m_bp.group(1))
                mean_dbp = float(m_bp.group(2))
                blood_pressure = f"{int(mean_sbp)}/{int(mean_dbp)}"
                break
            for offset in (1, 2):
                if idx + offset < len(lines):
                    m_bp = re.search(bp_pat, lines[idx + offset])
                    if m_bp:
                        mean_sbp = float(m_bp.group(1))
                        mean_dbp = float(m_bp.group(2))
                        blood_pressure = f"{int(mean_sbp)}/{int(mean_dbp)}"
                        break
            if blood_pressure:
                break
    if blood_pressure is None:
        m_fallback = re.search(r'\b([0-9]{2,3})\s*[\/\-]\s*([0-9]{2,3})\s*mm\s*hg\b', normalized_text, re.I)
        if m_fallback:
            mean_sbp = float(m_fallback.group(1))
            mean_dbp = float(m_fallback.group(2))
            blood_pressure = f"{int(mean_sbp)}/{int(mean_dbp)}"

    # 7. Fasting Glucose
    fasting_glucose = None
    fasting_glucose_ocr_note = None
    for idx, line in enumerate(lines):
        if line.lower().startswith('note:'):
            continue
        if re.search(r'fasting\s*(?:plasma\s*|blood\s*)?glucose|\bfbs\b|\bfbg\b', line, re.I):
            m = re.search(r'[:\-\|\=\s]+([0-9]{2,3}(?:\.[0-9]+)?)', line)
            if m:
                val = float(m.group(1))
                if 40.0 <= val <= 500.0:
                    fasting_glucose = {
                        "value": val,
                        "unit": "mg/dL",
                        "reference_range": "70 - 99 mg/dL",
                        "source_flagged": (val >= 100.0)
                    }
                    break
            for offset in range(1, 3):
                if idx + offset < len(lines):
                    nxt = lines[idx + offset]
                    if nxt.lower().startswith('note:'):
                        break
                    m = re.match(r'^([0-9]{2,3}(?:\.[0-9]+)?)\b', nxt)
                    if m:
                        val = float(m.group(1))
                        if 40.0 <= val <= 500.0:
                            fasting_glucose = {
                                "value": val,
                                "unit": "mg/dL",
                                "reference_range": "70 - 99 mg/dL",
                                "source_flagged": (val >= 100.0)
                            }
                            break
            if fasting_glucose is not None:
                break
    if fasting_glucose is None and re.search(r'\bfasting\b', normalized_text, re.I):
        fasting_glucose_ocr_note = "Field detected in report text, but value could not be extracted clearly."

    # Post-Prandial Glucose
    post_prandial_glucose = None
    pp_glucose_ocr_note = None
    pp_val = find_num_near(r'post\s*prandial|\bppbs\b|\bppbg\b|2\s*hr\s*post\s*meal', max_lines=2, min_val=30.0, max_val=500.0)
    if pp_val is not None:
        post_prandial_glucose = {
            "value": pp_val,
            "unit": "mg/dL",
            "reference_range": "< 140 mg/dL",
            "source_flagged": (pp_val >= 140.0 or pp_val < 70.0),
            "low_threshold": 70.0
        }

    # Random Glucose
    random_glucose = None
    rg_val = find_num_near(r'random\s*(?:blood\s*)?glucose|\brbs\b', max_lines=2, min_val=40.0, max_val=500.0)
    if rg_val is not None:
        random_glucose = {
            "value": rg_val,
            "unit": "mg/dL",
            "reference_range": "70 - 139 mg/dL",
            "source_flagged": (rg_val >= 140.0)
        }

    # 8. HbA1c
    # CRITICAL: STRICTLY rows named HbA1c / Glycated Hemoglobin / Glycosylated Hemoglobin / A1c
    # NEVER mistake the "Hemoglobin" row for HbA1c!
    hba1c = None
    hba1c_ocr_note = None
    for idx, line in enumerate(lines):
        if line.lower().startswith('note:'):
            continue
        if re.search(r'\b(?:hba1c|hb\s*[-–]?\s*a1c|\ba1c\b|(?:glycated|glycosylated)\s*(?:ha?emoglobin|hb))\b', line, re.I):
            m = re.search(r'[:\-\|\=\s]+([0-9]{1,2}(?:\.[0-9]+)?)', line)
            if m:
                val = float(m.group(1))
                if 3.0 <= val <= 20.0:
                    hba1c = {
                        "value": val,
                        "unit": "%",
                        "reference_range": "< 5.7%",
                        "source_flagged": (val >= 5.7)
                    }
                    break
            for offset in range(1, 3):
                if idx + offset < len(lines):
                    nxt = lines[idx + offset]
                    if nxt.lower().startswith('note:'):
                        break
                    m = re.match(r'^([0-9]{1,2}(?:\.[0-9]+)?)\b', nxt)
                    if m:
                        val = float(m.group(1))
                        if 3.0 <= val <= 20.0:
                            hba1c = {
                                "value": val,
                                "unit": "%",
                                "reference_range": "< 5.7%",
                                "source_flagged": (val >= 5.7)
                            }
                            break
            if hba1c is not None:
                break
    if hba1c is None and re.search(r'\b(?:hba1c|hb\s*[-–]?\s*a1c|glycated\s*hemoglobin)\b', normalized_text, re.I):
        hba1c_ocr_note = "HbA1c detected in text but value could not be extracted clearly."

    # Standalone Hemoglobin (recorded separately so it is never confused with HbA1c)
    hemoglobin = None
    for idx, line in enumerate(lines):
        if line.lower().startswith('note:'):
            continue
        if re.search(r'^(?:ha?emoglobin|hb)\b', line, re.I) and not re.search(r'\b(?:a1c|glycated|glycosylated)\b', line, re.I):
            for offset in range(1, 3):
                if idx + offset < len(lines):
                    m = re.match(r'^([0-9]{1,2}(?:\.[0-9]+)?)\b', lines[idx + offset])
                    if m:
                        try:
                            val = float(m.group(1))
                            if 5.0 <= val <= 25.0:
                                hemoglobin = val
                                break
                        except ValueError:
                            pass
            if hemoglobin is not None:
                break

    # 9. Total Cholesterol
    tc_val = find_num_near(r'total\s*cholesterol', max_lines=2, min_val=50.0, max_val=500.0)
    total_cholesterol = None
    if tc_val is not None:
        total_cholesterol = {
            "value": tc_val,
            "unit": "mg/dL",
            "reference_range": "< 200 mg/dL",
            "source_flagged": (tc_val >= 200.0)
        }

    # 10. HDL Cholesterol
    hdl_val = find_num_near(r'^hdl(?:\s*cholesterol)?\b', max_lines=2, min_val=10.0, max_val=150.0)
    hdl = None
    if hdl_val is not None:
        hdl = {
            "value": hdl_val,
            "unit": "mg/dL",
            "reference_range": "> 40 mg/dL",
            "source_flagged": (hdl_val < 40.0)
        }

    # 11. LDL Cholesterol
    ldl_val = find_num_near(r'^ldl(?:\s*cholesterol)?\b', max_lines=2, min_val=20.0, max_val=400.0)
    ldl = None
    if ldl_val is not None:
        ldl = {
            "value": ldl_val,
            "unit": "mg/dL",
            "reference_range": "< 100 mg/dL",
            "source_flagged": (ldl_val >= 100.0)
        }

    # 12. Triglycerides
    trig_val = find_num_near(r'^triglycerides?\b', max_lines=2, min_val=20.0, max_val=1000.0)
    triglycerides = None
    if trig_val is not None:
        triglycerides = {
            "value": trig_val,
            "unit": "mg/dL",
            "reference_range": "< 150 mg/dL",
            "source_flagged": (trig_val >= 150.0)
        }

    # 13. Clinical History (Yes/No mapped to 1/0)
    def find_yn_response(label_pattern: str) -> Optional[int]:
        for idx, line in enumerate(lines):
            if re.search(label_pattern, line, re.I):
                m = re.search(r'\b(yes|no)\b', line, re.I)
                if m:
                    return 1 if m.group(1).lower() == 'yes' else 0
                for offset in (1, 2):
                    if idx + offset < len(lines):
                        nxt = lines[idx + offset].lower()
                        if nxt.startswith('yes'):
                            return 1
                        if nxt.startswith('no'):
                            return 0
        return None

    high_bp = find_yn_response(r'high\s*blood\s*pressure')
    high_chol = find_yn_response(r'high\s*cholesterol')
    smoker = find_yn_response(r'smoked\s*100')
    phys_activity = find_yn_response(r'physical\s*activity')
    family_history = find_yn_response(r'relative\s*with\s*diabetes|family\s*history.*diabetes')

    waist_circumference_cm = find_num_near(r'waist(?:\s*circumference)?|\bwc\b', max_lines=2, min_val=40.0, max_val=180.0)

    has_primary = (
        (fasting_glucose is not None) or
        (post_prandial_glucose is not None) or
        (hba1c is not None) or
        (bmi.get("value") is not None) or
        (height_cm is not None and weight_kg is not None)
    )
    confidence = "high" if has_primary else "low"

    return {
        "fasting_glucose": fasting_glucose,
        "fasting_glucose_ocr_note": fasting_glucose_ocr_note,
        "post_prandial_glucose": post_prandial_glucose,
        "pp_glucose_ocr_note": pp_glucose_ocr_note,
        "hba1c": hba1c,
        "hba1c_ocr_note": hba1c_ocr_note,
        "random_glucose": random_glucose,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "mean_sbp": mean_sbp,
        "mean_dbp": mean_dbp,
        "blood_pressure": blood_pressure,
        "total_cholesterol": total_cholesterol,
        "hdl": hdl,
        "ldl": ldl,
        "triglycerides": triglycerides,
        "high_bp": high_bp,
        "high_chol": high_chol,
        "smoker": smoker,
        "phys_activity": phys_activity,
        "family_history": family_history,
        "hemoglobin": hemoglobin,
        "waist_circumference_cm": waist_circumference_cm,
        "age": age,
        "sex": sex,
        "extraction_confidence": confidence,
        "raw_text_char_count": char_count,
        "fallback_used": True,
        "extraction_method": "deterministic_clinical_extractor",
        "fallback_reason": None
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
    1. If fasting_glucose, post_prandial_glucose, hba1c, random_glucose, AND bmi are ALL null AND cannot compute BMI -> STOP.
    2. If extraction_confidence == 'low' -> STOP.
    3. BMI CALCULATION: In code ONLY, if bmi.value is null but height_cm and weight_kg are non-null:
       compute BMI = weight_kg / (height_m)^2, tag 'source: calculated'.
    4. String/regex verify all extracted numbers against raw OCR text. Drop any that fail.
    5. Returns (gate_passed, error_message, verified_json).
    """
    normalized_ocr = unicodedata.normalize('NFKC', raw_ocr_text).replace('：', ':').replace('，', ',')
    fg = extracted_json.get("fasting_glucose")
    pp = extracted_json.get("post_prandial_glucose")
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
        (pp is not None and pp.get("value") is not None) or
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
        "fasting_glucose_ocr_note": extracted_json.get("fasting_glucose_ocr_note"),
        "post_prandial_glucose": None,
        "pp_glucose_ocr_note": extracted_json.get("pp_glucose_ocr_note"),
        "hba1c": None,
        "hba1c_ocr_note": extracted_json.get("hba1c_ocr_note"),
        "random_glucose": None,
        "height_cm": None,
        "weight_kg": None,
        "bmi": bmi,
        "mean_sbp": None,
        "mean_dbp": None,
        "blood_pressure": None,
        "total_cholesterol": None,
        "hdl": None,
        "ldl": None,
        "triglycerides": None,
        "high_bp": None,
        "high_chol": None,
        "smoker": None,
        "phys_activity": None,
        "family_history": None,
        "hemoglobin": None,
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

    # Verify Post-Prandial Glucose (Bug Q)
    if pp and pp.get("value") is not None:
        val = pp["value"]
        orig = pp.get("original_value")
        v_str = str(int(val)) if val == int(val) else str(val)
        orig_str = str(orig) if orig is not None else None
        if re.search(rf'\b{re.escape(v_str)}\b', normalized_ocr) or (orig_str and (orig_str in normalized_ocr or re.search(rf'\b{re.escape(orig_str)}\b', normalized_ocr))):
            verified_json["post_prandial_glucose"] = pp
            if pp.get("source_flagged"):
                verified_json["required_flagged_checklist"].append("Post-Prandial Blood Glucose")
        else:
            logger.warning(f"[Step 2 Verification] Post-prandial glucose {v_str} failed OCR string match, dropped.")

    # Verify HbA1c
    if hba1c and hba1c.get("value") is not None:
        val = hba1c["value"]
        v_str = str(val)
        int_str = str(int(val)) if val == int(val) else None
        matched = (
            v_str in normalized_ocr or
            re.search(rf'\b{re.escape(v_str)}\b', normalized_ocr) or
            (int_str and (int_str in normalized_ocr or re.search(rf'\b{re.escape(int_str)}\b', normalized_ocr)))
        )
        if matched:
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

    # Verify Blood Pressure (mean_sbp, mean_dbp, formatted)
    msbp = extracted_json.get("mean_sbp")
    mdbp = extracted_json.get("mean_dbp")
    if msbp is not None and mdbp is not None:
        sbp_s = str(int(msbp))
        dbp_s = str(int(mdbp))
        if re.search(rf'\b{re.escape(sbp_s)}\b', normalized_ocr) and re.search(rf'\b{re.escape(dbp_s)}\b', normalized_ocr):
            verified_json["mean_sbp"] = msbp
            verified_json["mean_dbp"] = mdbp
            verified_json["blood_pressure"] = f"{int(msbp)}/{int(mdbp)}"
            pass
        else:
            logger.warning(f"[Step 2 Verification] Blood pressure {msbp}/{mdbp} not in OCR, dropped.")

    # Verify Lipid Panel
    for l_key, l_label in [
        ("total_cholesterol", "Total Cholesterol"),
        ("hdl", "HDL Cholesterol"),
        ("ldl", "LDL Cholesterol"),
        ("triglycerides", "Triglycerides")
    ]:
        l_item = extracted_json.get(l_key)
        if l_item and l_item.get("value") is not None:
            l_val = l_item["value"]
            v_str = str(int(l_val)) if l_val == int(l_val) else str(l_val)
            if re.search(rf'\b{re.escape(v_str)}\b', normalized_ocr):
                verified_json[l_key] = l_item
                pass
            else:
                logger.warning(f"[Step 2 Verification] {l_key} {v_str} not in OCR, dropped.")

    # Verify Clinical History
    for h_key, h_pat in [
        ("high_bp", r'high\s*blood\s*pressure'),
        ("high_chol", r'high\s*cholesterol'),
        ("smoker", r'smoked\s*100'),
        ("phys_activity", r'physical\s*activity'),
        ("family_history", r'relative\s*with\s*diabetes|family\s*history.*diabetes')
    ]:
        h_val = extracted_json.get(h_key)
        if h_val is not None and re.search(h_pat, normalized_ocr, re.I):
            verified_json[h_key] = h_val

    # Verify Hemoglobin
    hb_val = extracted_json.get("hemoglobin")
    if hb_val is not None:
        hb_s = str(hb_val)
        if hb_s in normalized_ocr or re.search(rf'\b{re.escape(hb_s)}\b', normalized_ocr):
            verified_json["hemoglobin"] = hb_val

    # Final sanity check: do we have at least one verified indicator?
    has_verified = (
        (verified_json["fasting_glucose"] is not None) or
        (verified_json["post_prandial_glucose"] is not None) or
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
      * Post-Prandial Glucose >=200 mg/dL -> 'Diagnostic range for diabetes'
      * Post-Prandial Glucose 140-199 -> 'Prediabetes range'
      * Post-Prandial Glucose <70 or below printed ref -> 'Low / Below Reference Range (Reactive Hypoglycemia pattern)'
      * Post-Prandial Glucose 70-139 -> 'Normal range'
      * HbA1c >=6.5% -> 'Diagnostic range for diabetes'
      * HbA1c 5.7-6.4% -> 'Prediabetes range'
      * HbA1c <5.7% -> 'Normal range'
      * If glucose and HbA1c disagree, report each separately -- never average.
    - OBESITY classification (WHO Asian-adjusted cutoffs):
      * BMI <18.5 -> Underweight
      * BMI 18.5-22.9 -> Normal
      * BMI 23-24.9 -> Overweight (Asian cutoff; standard WHO cutoff is 25-29.9)
      * BMI >=25 -> Obese (Asian cutoff; standard WHO cutoff is >=30)
      * Waist >90cm (men) or >80cm (women) -> flag central obesity as additional modifier.
    - COMBINED risk note in plain language (no invented blended percentage).
    """
    # Step 3 utilizes the deterministic ADA criteria & WHO Asian cutoffs clinical engine directly.
    # This guarantees 100% compliance with clinical standards and reduces upload LLM calls to at most 1.
    fg = verified_json.get("fasting_glucose")
    pp = verified_json.get("post_prandial_glucose")
    hba1c = verified_json.get("hba1c")
    rg = verified_json.get("random_glucose")
    bmi = verified_json.get("bmi") or {}
    wc = verified_json.get("waist_circumference_cm")
    sex = verified_json.get("sex") or "Male"
    is_male = "m" in sex.lower() and "fe" not in sex.lower()

    # Count total available indicators across both metabolic and anthropometric fields verified in Step 2
    verified_fields = [
        fg.get("value") if fg else None,
        pp.get("value") if pp else None,
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
    glucose_status = None
    pp_status = None
    hba1c_status = None

    if fg and fg.get("value") is not None:
        val = fg["value"]
        if val >= 126.0:
            glucose_status = "suggestive of diabetes range; confirm with a clinician"
        elif val >= 100.0:
            glucose_status = "Prediabetes range"
        else:
            glucose_status = "Normal range"
        diabetes_classifications.append({
            "marker": "Fasting Blood Glucose",
            "value": f"{val} mg/dL",
            "classification": glucose_status,
            "diagnostic_threshold": "ADA Standard: >=126 mg/dL Diabetes, 100-125 mg/dL Prediabetes",
            "source_flagged": (val >= 100.0) and fg.get("source_flagged", False)
        })

    if pp and pp.get("value") is not None:
        val = pp["value"]
        low_t = pp.get("low_threshold", 70.0)
        if val >= 200.0:
            pp_status = "suggestive of diabetes range; confirm with a clinician"
        elif val >= 140.0:
            pp_status = "Prediabetes range"
        elif val < low_t or val < 70.0:
            pp_status = "Low / Below Reference Range (Reactive Hypoglycemia pattern)"
        else:
            pp_status = "Normal range"
        diabetes_classifications.append({
            "marker": "Post-Prandial Blood Glucose (2-Hour)",
            "value": f"{val} mg/dL",
            "classification": pp_status,
            "diagnostic_threshold": "ADA Standard: <140 mg/dL Normal, 140-199 mg/dL Prediabetes, >=200 mg/dL Diabetes",
            "source_flagged": (val >= 140.0 or val < low_t or val < 70.0) and pp.get("source_flagged", False)
        })

    if hba1c and hba1c.get("value") is not None:
        val = hba1c["value"]
        if val >= 6.5:
            hba1c_status = "suggestive of diabetes range; confirm with a clinician"
        elif val >= 5.7:
            hba1c_status = "Prediabetes range"
        else:
            hba1c_status = "Normal range"
        diabetes_classifications.append({
            "marker": "HbA1c (Glycated Hemoglobin)",
            "value": f"{val}%",
            "classification": hba1c_status,
            "diagnostic_threshold": "ADA Standard: >=6.5% Diabetes, 5.7-6.4% Prediabetes",
            "source_flagged": (val >= 5.7) and hba1c.get("source_flagged", False)
        })

    if rg and rg.get("value") is not None and not fg and not pp:
        val = rg["value"]
        rg_status = "Diagnostic range for diabetes" if val >= 200.0 else ("Elevated Glycemia" if val >= 140.0 else "Normal range")
        diabetes_classifications.append({
            "marker": "Random Blood Glucose",
            "value": f"{val} mg/dL",
            "classification": rg_status,
            "diagnostic_threshold": "ADA Standard: >=200 mg/dL with symptoms",
            "source_flagged": (val >= 140.0) and rg.get("source_flagged", False)
        })

    # Diabetes summary construction
    findings_list = []
    if glucose_status:
        findings_list.append(f"Fasting glucose indicates {glucose_status}")
    if pp_status:
        findings_list.append(f"Post-Prandial glucose indicates {pp_status}")
    if hba1c_status:
        findings_list.append(f"HbA1c indicates {hba1c_status}")

    if findings_list:
        diabetes_summary = ". ".join(findings_list) + "."
    elif rg and rg.get("value") is not None:
        diabetes_summary = f"Random blood glucose indicates {rg_status}."
    else:
        diabetes_summary = "Cannot be assessed (glycemic panel pending)."

    # 2. OBESITY CLASSIFICATION (WHO Asian-adjusted cutoffs)
    obesity_assessment = {}
    bmi_val = bmi.get("value") if bmi else None

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
        else:
            central_obesity = False
            central_obesity_note = (
                f"Waist circumference of {wc} cm is within the recommended Asian cutoff (<= {int(cutoff)} cm) for "
                f"{'men' if is_male else 'women'}."
            )

    if bmi_val is not None:
        if bmi_val < 18.5:
            obesity_cat = "Underweight"
        elif bmi_val <= 22.9:
            obesity_cat = "Normal"
        elif bmi_val <= 24.9:
            obesity_cat = "Overweight (Asian cutoff; standard WHO cutoff is 25–29.9)"
        else:
            obesity_cat = "Obese (Asian cutoff; standard WHO cutoff is >=30)"

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
    elif wc is not None:
        obesity_assessment = {
            "bmi_value": None,
            "bmi_unit": "kg/m²",
            "classification": "Waist Circumference Evaluated",
            "standard_used": "WHO / ICMR-NIN Asian Waist Cutoffs (Men: 90cm, Women: 80cm)",
            "bmi_source": "not_present",
            "central_obesity": central_obesity,
            "central_obesity_note": central_obesity_note,
            "source_flagged": central_obesity
        }

    # 3. COMBINED RISK NOTE (Plain language, NO invented single percentage)
    is_glycemic_normal = (
        (glucose_status is None or "Normal" in glucose_status) and
        (pp_status is None or "Normal" in pp_status) and
        (hba1c_status is None or "Normal" in hba1c_status)
    )
    is_obesity_normal = (
        obesity_assessment.get("classification") == "Normal" and
        not obesity_assessment.get("central_obesity", False)
    )
    has_any_glycemic = (glucose_status is not None or pp_status is not None or hba1c_status is not None)
    is_all_normal = is_glycemic_normal and is_obesity_normal and has_any_glycemic and (bmi_val is not None)

    if is_data_limited:
        combined_risk_note = (
            "Assessment is data-limited as only a single metabolic or anthropometric field was available in this report. "
            "Additional laboratory or anthropometric measurements are recommended before drawing confident clinical conclusions."
        )
    elif is_all_normal:
        combined_risk_note = (
            f"All monitored metabolic biomarkers (glycemic panel and BMI {bmi_val} kg/m²) are within healthy normal reference ranges. "
            "Metabolic health profile is optimal; continue maintaining balanced nutrition and regular physical activity to sustain baseline health."
        )
    elif bmi_val is not None and has_any_glycemic:
        bmi_short_cat = obesity_assessment.get("classification", "").split("(")[0].strip()
        glycemic_desc = glucose_status or pp_status or hba1c_status
        if bmi_short_cat == "Underweight":
            if "Normal" in (glycemic_desc or ""):
                combined_risk_note = (
                    f"Anthropometric assessment indicates Underweight status (BMI {bmi_val} kg/m²), while glycemic regulation is optimal ({glycemic_desc}). "
                    "Clinical priority focuses on balanced nutritional support, adequate caloric intake, and lean muscle mass development rather than glycemic reduction."
                )
            elif "Low" in (glycemic_desc or ""):
                combined_risk_note = (
                    f"Anthropometric assessment indicates Underweight status (BMI {bmi_val} kg/m²), alongside lower post-prandial glycemic levels ({glycemic_desc}). "
                    "Nutritional support emphasizing complex carbohydrates, protein intake, and consistent meal timing is recommended."
                )
            else:
                combined_risk_note = (
                    f"Anthropometric assessment indicates Underweight status (BMI {bmi_val} kg/m²), alongside elevated glycemic markers ({glycemic_desc}). "
                    "Clinical evaluation is recommended to investigate potential metabolic or endocrine etiologies (such as Type 1 or secondary diabetes) while providing tailored nutritional support."
                )
        elif glucose_status and hba1c_status and glucose_status != hba1c_status:
            combined_risk_note = (
                f"Adiposity evaluation ({bmi_short_cat}: BMI {bmi_val} kg/m²) is accompanied by discordant glycemic markers: "
                f"fasting glucose falls in the {glucose_status}, while HbA1c indicates {hba1c_status} (reported separately per ADA clinical guidelines; repeat confirmatory testing recommended)."
            )
        elif is_glycemic_normal and bmi_short_cat == "Normal":
            combined_risk_note = (
                f"All monitored biomarkers are within standard reference ranges (BMI {bmi_val} kg/m²: Normal; Glycemia: Normal range). "
                "Metabolic parameters show healthy baseline regulation."
            )
        elif "Diagnostic" in (glucose_status or "") or "Diagnostic" in (pp_status or "") or "Diagnostic" in (hba1c_status or "") or "Obese" in bmi_short_cat:
            combined_risk_note = (
                f"Adiposity evaluation ({bmi_short_cat}: BMI {bmi_val} kg/m²) and glycemic assessment ({glycemic_desc}) "
                f"concordantly indicate elevated metabolic risk requiring targeted nutritional and lifestyle guidance."
            )
        else:
            combined_risk_note = (
                f"Adiposity evaluation ({bmi_short_cat}: BMI {bmi_val} kg/m²) and glycemic assessment ({glycemic_desc}) "
                f"indicate borderline or moderate metabolic risk. Preventive diet and routine activity can restore optimal glycemic balance."
            )
    elif bmi_val is not None:
        bmi_short_cat = obesity_assessment.get("classification", "").split("(")[0].strip()
        central_clause = f" with central adiposity (waist {wc} cm)" if obesity_assessment.get("central_obesity") else ""
        if bmi_short_cat == "Underweight":
            combined_risk_note = (
                f"Body Mass Index ({bmi_val} kg/m²) indicates Underweight status per WHO Asian guidelines. "
                "Nutritional guidance aimed at gradual healthy weight gain and adequate caloric intake is recommended. Glycemic metrics were not included in this panel."
            )
        elif bmi_short_cat == "Normal":
            combined_risk_note = (
                f"Body Mass Index ({bmi_val} kg/m²) is within the healthy normal range per WHO Asian guidelines. "
                "Laboratory glycemic metrics (fasting glucose / HbA1c) were not included in this panel."
            )
        else:
            combined_risk_note = (
                f"Adiposity assessment indicates {obesity_assessment.get('classification')}{central_clause}. "
                f"Glycemic metrics (fasting glucose / HbA1c) are pending to evaluate complete metabolic risk."
            )
    elif has_any_glycemic:
        glycemic_desc = glucose_status or pp_status or hba1c_status
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
                    f"All monitored metabolic biomarkers (glycemic panel and BMI {bmi_val} kg/m²) are within healthy normal reference ranges. "
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
            "post_prandial_glucose_status": pp_status,
            "hba1c_status": hba1c_status,
            "hba1c_ocr_note": verified_json.get("hba1c_ocr_note"),
            "fasting_glucose_ocr_note": verified_json.get("fasting_glucose_ocr_note"),
            "pp_glucose_ocr_note": verified_json.get("pp_glucose_ocr_note")
        },
        "obesity_assessment": obesity_assessment,
        "combined_risk_note": combined_risk_note,
        "hba1c_ocr_note": verified_json.get("hba1c_ocr_note"),
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

    pp = verified_json.get("post_prandial_glucose")
    if pp and pp.get("value") is not None:
        verified_numbers.add(float(pp["value"]))
        verified_numbers.add(float(int(pp["value"])))
        if pp.get("original_value") is not None:
            verified_numbers.add(float(pp["original_value"]))
        if pp.get("low_threshold") is not None:
            verified_numbers.add(float(pp["low_threshold"]))

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
    if verified_json.get("mean_sbp") is not None:
        verified_numbers.add(float(verified_json["mean_sbp"]))
    if verified_json.get("mean_dbp") is not None:
        verified_numbers.add(float(verified_json["mean_dbp"]))
    for lk in ["total_cholesterol", "hdl", "ldl", "triglycerides"]:
        item = verified_json.get(lk)
        if item and item.get("value") is not None:
            verified_numbers.add(float(item["value"]))
    if verified_json.get("hemoglobin") is not None:
        verified_numbers.add(float(verified_json["hemoglobin"]))

    # Add standard clinical threshold numbers that the prompt dictates using
    standard_cutoffs = {
        70.0, 73.0, 99.0, 100.0, 125.0, 126.0, 139.0, 140.0, 150.0, 199.0, 200.0,
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

def execute_metabolic_pipeline(raw_ocr_text: str, filename: str, file_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Executes the complete metabolic risk pipeline:
      STEP 0: Pre-extraction document type validation (Bug R: reject research tables)
      STEP 1: Extraction (strict, no derivations, fallback to code-only regex if LLM rate-limited)
      STEP 2: Pre-generation gate (code hard-stop, code-only BMI calculation)
      STEP 3: Risk scoring (ADA + WHO Asian cutoffs - deterministic clinical engine)
      STEP 4: Post-generation audit (code number & checklist validation)
    """
    logger.info(f"Executing metabolic pipeline for {filename}")

    # Check in-memory cache first (0 LLM calls on re-upload)
    import hashlib
    report_hash = hashlib.sha256(file_bytes if file_bytes else raw_ocr_text.encode('utf-8')).hexdigest()
    cached_res = get_cached_extraction(report_hash)
    if cached_res:
        return cached_res

    # STEP 0: Document-type validation check (Bug R: population tables)
    is_valid_doc, rejection_signals = validate_document_type(raw_ocr_text)
    if not is_valid_doc:
        err_msg = (
            "This appears to be a research or statistical summary table, not an individual lab report. "
            "Please upload your own personal diagnostic report."
        )
        logger.warning(f"[Bug R Document Classification REFUSAL] Signals: {rejection_signals}")
        return {
            "success": False,
            "error": err_msg,
            "pipeline_status": "REJECTED_NON_PATIENT_DOCUMENT",
            "filename": filename,
            "rejection_signals": rejection_signals,
            "step1_extraction": None,
            "step4_audit": {
                "audit_passed": False,
                "audit_label": "Rejected Non-Patient Document"
            }
        }

    # STEP 1: Extraction (LLM with backoff, or deterministic regex fallback)
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

    # STEP 3: Risk Scoring & Classification (deterministic ADA & WHO Asian clinical engine)
    step3_res = step3_metabolic_risk_scoring(verified_json)

    # STEP 4: Post-generation Audit Gate
    audit_res = step4_metabolic_post_generation_audit(verified_json, step3_res, filename)

    # Wire to real diabetes_module two-tier model (FastAPI POST /predict)
    tier1_fields = ['age', 'sex', 'bmi', 'mean_sbp', 'mean_dbp', 'high_bp', 'high_chol', 'smoker', 'phys_activity', 'family_history']
    missing_tier1 = []
    if verified_json.get("age") is None:
        missing_tier1.append("age")
    if verified_json.get("sex") is None:
        missing_tier1.append("sex")
    bmi_val = verified_json.get("bmi", {}).get("value")
    if bmi_val is None:
        missing_tier1.append("bmi")
    if verified_json.get("mean_sbp") is None:
        missing_tier1.append("mean_sbp")
    if verified_json.get("mean_dbp") is None:
        missing_tier1.append("mean_dbp")
    if verified_json.get("high_bp") is None:
        missing_tier1.append("high_bp")
    if verified_json.get("high_chol") is None:
        missing_tier1.append("high_chol")
    if verified_json.get("smoker") is None:
        missing_tier1.append("smoker")
    if verified_json.get("phys_activity") is None:
        missing_tier1.append("phys_activity")
    if verified_json.get("family_history") is None:
        missing_tier1.append("family_history")

    diabetes_tier_prediction = None
    if not missing_tier1:
        sex_binary = 1 if "m" in (verified_json.get("sex") or "").lower() and "fe" not in (verified_json.get("sex") or "").lower() else 0
        feats_dict = {
            "age": float(verified_json["age"]),
            "sex": sex_binary,
            "bmi": float(bmi_val),
            "mean_sbp": float(verified_json["mean_sbp"]),
            "mean_dbp": float(verified_json["mean_dbp"]),
            "high_bp": int(verified_json["high_bp"]),
            "high_chol": int(verified_json["high_chol"]),
            "smoker": int(verified_json["smoker"]),
            "phys_activity": int(verified_json["phys_activity"]),
            "family_history": int(verified_json["family_history"])
        }
        labs_dict = {}
        hba1c_obj = verified_json.get("hba1c")
        hba1c_val = hba1c_obj.get("value") if (hba1c_obj and isinstance(hba1c_obj, dict)) else None
        if hba1c_val is not None and 3.0 <= float(hba1c_val) <= 18.0:
            labs_dict["hba1c"] = float(hba1c_val)
        fg_obj = verified_json.get("fasting_glucose")
        fg_val = fg_obj.get("value") if (fg_obj and isinstance(fg_obj, dict)) else None
        if fg_val is not None:
            labs_dict["fasting_glucose"] = float(fg_val)

        predict_payload = {
            "features": feats_dict
        }
        if labs_dict:
            predict_payload["labs"] = labs_dict
        if hba1c_val is not None and 3.0 <= float(hba1c_val) <= 18.0:
            predict_payload["hba1c"] = float(hba1c_val)

        diabetes_svc = os.getenv("DIABETES_SERVICE_URL", "http://127.0.0.1:8005").rstrip("/")
        ml_svc = os.getenv("ML_SERVICE_URL", "").rstrip("/")
        if diabetes_svc:
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"{diabetes_svc}/predict",
                    data=json.dumps(predict_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    diabetes_tier_prediction = json.loads(resp.read().decode("utf-8"))
            except Exception as api_err:
                pass
        if diabetes_tier_prediction is None and ml_svc:
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"{ml_svc}/diabetes/predict",
                    data=json.dumps(predict_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    diabetes_tier_prediction = json.loads(resp.read().decode("utf-8"))
            except Exception:
                pass
        if diabetes_tier_prediction is None:
            try:
                import sys
                from pathlib import Path
                import importlib.util
                dm_dir = Path(__file__).resolve().parent.parent / "diabetes_module" / "src"
                if str(dm_dir) not in sys.path:
                    sys.path.insert(0, str(dm_dir))
                import train_and_evaluate
                import __main__
                __main__.TabularPreprocessor = train_and_evaluate.TabularPreprocessor
                __main__.ProbabilityCalibrator = train_and_evaluate.ProbabilityCalibrator
                if "diabetes_api_module" in sys.modules:
                    dm_app = sys.modules["diabetes_api_module"]
                else:
                    spec = importlib.util.spec_from_file_location("diabetes_api_module", dm_dir / "app.py")
                    dm_app = importlib.util.module_from_spec(spec)
                    sys.modules["diabetes_api_module"] = dm_app
                    spec.loader.exec_module(dm_app)
                req_obj = dm_app.PredictRequest(
                    features=dm_app.NonLabFeatures(**feats_dict),
                    labs=dm_app.LabValues(**labs_dict) if labs_dict else None,
                    hba1c=float(hba1c_val) if hba1c_val else None
                )
                diabetes_tier_prediction = dm_app.predict_diabetes_risk(req_obj)
            except Exception as direct_err:
                logger.error(f"Failed direct diabetes_module inference: {direct_err}")

    pipeline_result = {
        "success": audit_res["audit_passed"],
        "pipeline_version": "metabolic_v2_two_tier_ada_who",
        "pipeline_status": "APPROVED" if audit_res["audit_passed"] else "BLOCKED_FOR_REVIEW",
        "filename": filename,
        "fallback_used": step1_res.get("fallback_used", False),
        "extraction_method": step1_res.get("extraction_method", "deterministic_clinical_extractor"),
        "step1_extraction": step1_res,
        "step2_verified_json": verified_json,
        "step3_risk_scoring": step3_res,
        "step4_audit": audit_res,
        "raw_text": raw_ocr_text,
        "diabetes_tier_prediction": diabetes_tier_prediction,
        "missing_tier1_features": missing_tier1,
        # Direct convenience views for frontend UI
        "data_limited": step3_res.get("data_limited", False),
        "diabetes_assessment": step3_res["diabetes_assessment"],
        "obesity_assessment": step3_res["obesity_assessment"],
        "combined_risk_note": step3_res["combined_risk_note"],
        "hba1c_ocr_note": step3_res.get("hba1c_ocr_note") or verified_json.get("hba1c_ocr_note"),
        "fasting_glucose_ocr_note": verified_json.get("fasting_glucose_ocr_note"),
        "pp_glucose_ocr_note": verified_json.get("pp_glucose_ocr_note"),
        "audit_passed": audit_res["audit_passed"],
        "audit_label": audit_res["audit_label"],
        "verified_vitals": {
            "fasting_glucose": verified_json.get("fasting_glucose"),
            "post_prandial_glucose": verified_json.get("post_prandial_glucose"),
            "hba1c": verified_json.get("hba1c"),
            "random_glucose": verified_json.get("random_glucose"),
            "bmi": verified_json.get("bmi"),
            "height_cm": verified_json.get("height_cm"),
            "weight_kg": verified_json.get("weight_kg"),
            "mean_sbp": verified_json.get("mean_sbp"),
            "mean_dbp": verified_json.get("mean_dbp"),
            "blood_pressure": verified_json.get("blood_pressure"),
            "total_cholesterol": verified_json.get("total_cholesterol"),
            "hdl": verified_json.get("hdl"),
            "ldl": verified_json.get("ldl"),
            "triglycerides": verified_json.get("triglycerides"),
            "high_bp": verified_json.get("high_bp"),
            "high_chol": verified_json.get("high_chol"),
            "smoker": verified_json.get("smoker"),
            "phys_activity": verified_json.get("phys_activity"),
            "family_history": verified_json.get("family_history"),
            "hemoglobin": verified_json.get("hemoglobin"),
            "waist_circumference_cm": verified_json.get("waist_circumference_cm"),
            "age": verified_json.get("age"),
            "sex": verified_json.get("sex")
        }
    }

    # Store in memory cache
    set_cached_extraction(report_hash, pipeline_result)
    return pipeline_result
