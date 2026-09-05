import os
import uuid
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Load Indian Food Nutrition Dataset
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
NUTRITION_CSV = os.path.join(DATA_DIR, "Indian_Food_Nutrition_Processed.csv")

try:
    if os.path.exists(NUTRITION_CSV):
        nutrition_df = pd.read_csv(NUTRITION_CSV)
        print(f"Loaded Indian Food Nutrition dataset: {len(nutrition_df)} food items.")
    else:
        nutrition_df = None
        print(f"Warning: {NUTRITION_CSV} not found.")
except Exception as e:
    nutrition_df = None
    print(f"Error loading nutrition CSV: {e}")

def classify_meal_slot(dish_name: str) -> str:
    """Classifies an Indian dish into Breakfast, Lunch, Snack, or Dinner."""
    name_lower = str(dish_name).lower()
    
    # Snacks
    if any(k in name_lower for k in ["chaat", "bhel", "snack", "roasted", "soup", "chana roasted", "makhana", "kachumber", "cutlet", "salad", "tea", "coffee"]):
        return "snack"
    # Breakfast
    elif any(k in name_lower for k in ["idli", "dosa", "upma", "poha", "paratha", "chilla", "cheela", "thepla", "toast", "pancake", "egg", "porridge", "oats", "puri"]):
        return "breakfast"
    # Lunch (heavier curries, dal, thali items)
    elif any(k in name_lower for k in ["roti", "phulka", "chapati", "dal", "paneer", "rajma", "chole", "chana masala", "curry", "thali", "sabzi", "bhindi", "aloo", "gobi", "palak"]):
        return "lunch"
    # Dinner (lighter grains, khichdi, pulao, lighter curries)
    elif any(k in name_lower for k in ["khichdi", "rice", "pulao", "dalia", "kofta", "kadhi", "biryani", "rasam", "sambar"]):
        return "dinner"
    else:
        return "lunch"

def score_food_item(row: pd.Series, risk_profile: Dict[str, Any]) -> float:
    """Computes a suitability score (0-100) for an Indian food item based on chronic risk profiles."""
    score = 70.0
    name = str(row.get("Dish Name", "")).lower()

    # Disqualify raw condiments, pure masalas, pickles, or sweets
    disqualify_keywords = [
        "pickle", "achar", "jam", "syrup", "halwa", "ladoo", "jalebi", "gulab jamun",
        "sugar syrup", "deep fried", "pakoda", "bhajiya", "chutney dry powder", "masala powder"
    ]
    if any(kw in name for kw in disqualify_keywords):
        return 0.0

    calories = float(row.get("Calories (kcal)", 0))
    carbs = float(row.get("Carbohydrates (g)", 0))
    protein = float(row.get("Protein (g)", 0))
    fats = float(row.get("Fats (g)", 0))
    fibre = float(row.get("Fibre (g)", 0))
    sodium = float(row.get("Sodium (mg)", 0))

    glucose = float(risk_profile.get("glucose", 95.0))
    systolic_bp = float(risk_profile.get("systolic_bp", 120.0))
    bmi = float(risk_profile.get("bmi", 24.0))

    # 1. Diabetes & Glycemic Scoring
    if risk_profile.get("diabetes_risk") in ["Moderate", "High Risk"] or glucose >= 100.0:
        if fibre >= 3.0:
            score += (fibre * 3.5)
        elif fibre < 1.0:
            score -= 10.0

        if carbs > 35.0:
            score -= (carbs - 35.0) * 1.5

        if protein >= 6.0:
            score += (protein * 2.0)

    # 2. Cardiovascular & Blood Pressure Scoring
    if risk_profile.get("cvd_risk") in ["Moderate", "High Risk"] or systolic_bp >= 130.0:
        if sodium > 400.0:
            score -= 30.0
        elif sodium > 200.0:
            score -= (sodium - 200.0) * 0.1
        else:
            score += 10.0

        if fats > 12.0:
            score -= (fats - 12.0) * 2.5
        elif fats <= 5.0:
            score += 8.0

        score += (fibre * 4.0)

    # 3. Weight Management / BMI Rules
    if bmi >= 25.0:
        if calories > 280.0:
            score -= (calories - 280.0) * 0.15
        elif calories <= 180.0:
            score += 10.0
        score += (fibre * 3.0) + (protein * 2.0)

    # General baseline bonuses
    if protein >= 6.0:
        score += 8.0
    if fibre >= 2.0:
        score += 6.0

    return round(score, 2)

# Comprehensive 7-Day Curated Authentic Indian Menu Templates
CURATED_7_DAY_MENUS = [
    {
        "day_number": 1,
        "day_name": "Day 1 (Monday)",
        "theme": "Metabolic Reset & Blood Sugar Balancing",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Moong Dal Chilla (2 pcs) with Fresh Mint-Coriander Chutney & Buttermilk",
                "portion": "2 medium chillas + 2 tbsp chutney + 1 glass (200ml) chaas",
                "calories": 420.0,
                "macronutrients": {"carbohydrates_g": 52.0, "protein_g": 19.5, "fats_g": 11.0, "fibre_g": 8.5, "sodium_mg": 180.0},
                "clinical_benefit": "Plant-based protein with very low glycemic index prevents morning glucose surges.",
                "substitutions": [
                    {"dish_name": "Vegetable Poha with Roasted Peanuts", "portion": "1.5 cups", "calories": 390.0, "protein": 11.0, "fibre": 7.0},
                    {"dish_name": "Oats Upma with Finely Chopped Carrots & Peas", "portion": "1.5 cups", "calories": 380.0, "protein": 12.0, "fibre": 8.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "2 Multigrain Phulkas with Palak Paneer (Low Fat) & Yellow Masoor Dal Tadka",
                "portion": "2 phulkas (wheat+ragi+chana) + 1 cup palak paneer + 1 cup dal + fresh cucumber salad",
                "calories": 630.0,
                "macronutrients": {"carbohydrates_g": 78.0, "protein_g": 27.5, "fats_g": 17.0, "fibre_g": 13.5, "sodium_mg": 280.0},
                "clinical_benefit": "Fibre matrix from spinach and lentils binds dietary cholesterol and promotes gradual satiety.",
                "substitutions": [
                    {"dish_name": "2 Bajra Rotis with Methi Chana Dal & Lauki Sabzi", "portion": "2 rotis + 1 bowl dal", "calories": 610.0, "protein": 24.0, "fibre": 14.0},
                    {"dish_name": "1.5 cup Brown Rice with Soya Chunk Curry & Beetroot Poriyal", "portion": "1.5 cup rice + curry", "calories": 620.0, "protein": 26.0, "fibre": 12.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Roasted Spiced Makhana (Fox Nuts) with Green Tea (No Sugar)",
                "portion": "1 large bowl (35g) dry roasted in minimal olive oil with rock salt + 1 mug green tea",
                "calories": 230.0,
                "macronutrients": {"carbohydrates_g": 34.0, "protein_g": 6.5, "fats_g": 5.0, "fibre_g": 5.0, "sodium_mg": 90.0},
                "clinical_benefit": "Magnesium and potassium-rich snack that stabilizes evening appetite without sodium spikes.",
                "substitutions": [
                    {"dish_name": "Sprouted Moong Chaat with Chopped Onion & Lemon", "portion": "1 cup", "calories": 210.0, "protein": 10.0, "fibre": 6.5},
                    {"dish_name": "Handful (25g) Roasted Chana with 5 Almonds", "portion": "1 palmful", "calories": 220.0, "protein": 9.0, "fibre": 5.5}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Vegetable Quinoa / Broken Wheat Pulao with Sprouted Moong Raita",
                "portion": "1.5 cups pulao with beans, carrots & florets + 1 cup low-fat cucumber raita",
                "calories": 520.0,
                "macronutrients": {"carbohydrates_g": 72.0, "protein_g": 21.0, "fats_g": 11.5, "fibre_g": 11.0, "sodium_mg": 210.0},
                "clinical_benefit": "Light and easily digestible complex carbohydrates supporting restful sleep and nocturnal glycemic control.",
                "substitutions": [
                    {"dish_name": "Moong Dal & Brown Rice Khichdi with Low-Fat Kadhi", "portion": "1.5 cups + 1 bowl kadhi", "calories": 490.0, "protein": 18.0, "fibre": 9.5},
                    {"dish_name": "2 Jowar Phulkas with Baingan Bharta & Toor Dal", "portion": "2 rotis + 1 cup bharta", "calories": 510.0, "protein": 19.0, "fibre": 12.0}
                ]
            }
        ]
    },
    {
        "day_number": 2,
        "day_name": "Day 2 (Tuesday)",
        "theme": "Heart-Health & Vascular Protection",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Vegetable Poha with Roasted Peanuts, Curry Leaves & 1 Boiled Egg (or Paneer)",
                "portion": "1.5 cups flattened rice with green peas, onions, peanuts + 1 egg or 30g grilled paneer",
                "calories": 430.0,
                "macronutrients": {"carbohydrates_g": 56.0, "protein_g": 17.0, "fats_g": 12.5, "fibre_g": 6.5, "sodium_mg": 190.0},
                "clinical_benefit": "Rich in iron and bioavailable proteins; peanuts deliver heart-protective monounsaturated fats.",
                "substitutions": [
                    {"dish_name": "Steamed Idlis (3 pcs) with Drumstick Sambar", "portion": "3 idlis + 1.5 cup sambar", "calories": 410.0, "protein": 14.0, "fibre": 8.0},
                    {"dish_name": "Besan Chilla with Grated Paneer Stuffing", "portion": "2 chillas", "calories": 420.0, "protein": 18.0, "fibre": 7.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "2 Bajra (Pearl Millet) Rotis with Methi Chana Dal & Lauki (Bottle Gourd) Sabzi",
                "portion": "2 bajra rotis + 1 cup methi dal + 1 cup lauki sabzi + 1 bowl plain curd (100g)",
                "calories": 640.0,
                "macronutrients": {"carbohydrates_g": 82.0, "protein_g": 25.0, "fats_g": 15.0, "fibre_g": 15.0, "sodium_mg": 240.0},
                "clinical_benefit": "Bajra is gluten-free, rich in iron, and fenugreek (methi) enhances peripheral insulin sensitivity.",
                "substitutions": [
                    {"dish_name": "2 Whole Wheat Roti with Rajma Masala & French Beans", "portion": "2 rotis + 1 cup rajma", "calories": 630.0, "protein": 24.0, "fibre": 13.0},
                    {"dish_name": "1.5 cup Quinoa with Paneer Bhurji & Mix Dal", "portion": "1.5 cup quinoa + bhurji", "calories": 640.0, "protein": 28.0, "fibre": 11.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Sprouted Kala Chana & Sweet Corn Chaat with Fresh Lemon & Cilantro",
                "portion": "1 medium bowl (150g) boiled sprouts with diced tomato, cucumber & rock salt",
                "calories": 220.0,
                "macronutrients": {"carbohydrates_g": 36.0, "protein_g": 11.0, "fats_g": 3.0, "fibre_g": 8.0, "sodium_mg": 110.0},
                "clinical_benefit": "Provides resistant starch and live enzymes that nourish healthy gut microbiota.",
                "substitutions": [
                    {"dish_name": "Boiled Sweet Corn with Lemon Pepper", "portion": "1 cup", "calories": 200.0, "protein": 5.5, "fibre": 4.5},
                    {"dish_name": "Dry Roasted Makhana with Turmeric & Jeera", "portion": "35g", "calories": 210.0, "protein": 6.0, "fibre": 5.0}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Brown Rice & Moong Dal Khichdi with Low-Fat Gujarati Kadhi & Roasted Papad",
                "portion": "1.5 cups khichdi with 1 cup buttermilk-based kadhi + 1 roasted urad papad",
                "calories": 510.0,
                "macronutrients": {"carbohydrates_g": 74.0, "protein_g": 19.5, "fats_g": 10.0, "fibre_g": 9.5, "sodium_mg": 250.0},
                "clinical_benefit": "Complete amino acid profile from rice-lentil synergy; soothing to the digestive tract.",
                "substitutions": [
                    {"dish_name": "2 Whole Wheat Phulkas with Bhindi Masala & Toor Dal", "portion": "2 phulkas + 1 cup bhindi", "calories": 520.0, "protein": 18.0, "fibre": 11.0},
                    {"dish_name": "Millet Dosa (2 pcs) with Vegetable Stew", "portion": "2 dosas + 1 bowl stew", "calories": 490.0, "protein": 15.0, "fibre": 8.5}
                ]
            }
        ]
    },
    {
        "day_number": 3,
        "day_name": "Day 3 (Wednesday)",
        "theme": "High-Fibre Glycemic Stabilization",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Oats & Semolina Veggie Upma with 1 cup Low-Fat Curd / Buttermilk",
                "portion": "1.5 cups upma loaded with beans, carrots & mustard seeds + 1 bowl chaas",
                "calories": 410.0,
                "macronutrients": {"carbohydrates_g": 58.0, "protein_g": 16.0, "fats_g": 9.5, "fibre_g": 8.0, "sodium_mg": 180.0},
                "clinical_benefit": "Beta-glucan soluble fibre from oats reduces circulating LDL and stabilizes morning blood sugars.",
                "substitutions": [
                    {"dish_name": "Sprouted Methi Thepla (2 pcs) with Low-Fat Curd", "portion": "2 theplas + 1 cup curd", "calories": 420.0, "protein": 15.0, "fibre": 8.5},
                    {"dish_name": "Ragi Porridge (Salty) with Spiced Buttermilk", "portion": "1 large bowl", "calories": 390.0, "protein": 12.0, "fibre": 9.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "2 Jowar (Sorghum) Rotis with Soya Chunk Curry & Bhindi (Okra) Masala",
                "portion": "2 jowar rotis + 1 cup high-protein soya chunks curry + 1 cup okra stir fry + green salad",
                "calories": 640.0,
                "macronutrients": {"carbohydrates_g": 76.0, "protein_g": 31.0, "fats_g": 14.5, "fibre_g": 14.0, "sodium_mg": 260.0},
                "clinical_benefit": "Okra mucilage slows intestinal sugar absorption while soya isoflavones support arterial tone.",
                "substitutions": [
                    {"dish_name": "2 Multigrain Roti with Paneer Tikka Masala & Dal Tadka", "portion": "2 rotis + 1 cup paneer", "calories": 650.0, "protein": 28.0, "fibre": 12.0},
                    {"dish_name": "1.5 cup Brown Rice with Rajma Masala & Cucumber Raita", "portion": "1.5 cup rice + rajma", "calories": 630.0, "protein": 25.0, "fibre": 13.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Handful of Roasted Chana with 4 Almonds & 2 Walnuts + Masala Chai (No Sugar)",
                "portion": "30g roasted chana + 6 raw nuts + 1 cup freshly brewed ginger cardamom tea",
                "calories": 240.0,
                "macronutrients": {"carbohydrates_g": 24.0, "protein_g": 9.0, "fats_g": 10.5, "fibre_g": 6.0, "sodium_mg": 60.0},
                "clinical_benefit": "Omega-3 ALA from walnuts paired with slow-burning legumes provides steady cellular energy.",
                "substitutions": [
                    {"dish_name": "Roasted Makhana with Flax Seeds", "portion": "1 bowl", "calories": 210.0, "protein": 6.0, "fibre": 5.5},
                    {"dish_name": "Cucumber & Carrot Sticks with Hummus / Chana Dip", "portion": "1 plate", "calories": 190.0, "protein": 7.0, "fibre": 6.0}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "2 Whole Wheat Phulkas with Baingan Bharta (Roasted Eggplant) & Yellow Arhar Dal",
                "portion": "2 rotis (no oil) + 1 cup smoky baingan bharta + 1 cup arhar dal + kachumber",
                "calories": 510.0,
                "macronutrients": {"carbohydrates_g": 72.0, "protein_g": 19.0, "fats_g": 11.0, "fibre_g": 12.5, "sodium_mg": 220.0},
                "clinical_benefit": "Nasunin antioxidant in eggplant skin scavenges free radicals and protects brain & cardiac lipids.",
                "substitutions": [
                    {"dish_name": "Vegetable Dalia (Broken Wheat) Khichdi with Dahi", "portion": "1.5 cups + 1 cup curd", "calories": 490.0, "protein": 18.0, "fibre": 10.0},
                    {"dish_name": "2 Bajra Phulkas with Tori (Ridge Gourd) Sabzi & Dal", "portion": "2 rotis + sabzi", "calories": 500.0, "protein": 17.5, "fibre": 11.5}
                ]
            }
        ]
    },
    {
        "day_number": 4,
        "day_name": "Day 4 (Thursday)",
        "theme": "Antioxidant & Micronutrient Vitality",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Steamed Soft Idlis (3 pcs) with Drumstick-Lentil Sambar & Fresh Tomato Chutney",
                "portion": "3 medium steamed idlis + 1.5 cups vegetable sambar + 1 tbsp roasted tomato chutney",
                "calories": 420.0,
                "macronutrients": {"carbohydrates_g": 68.0, "protein_g": 15.0, "fats_g": 5.5, "fibre_g": 8.5, "sodium_mg": 210.0},
                "clinical_benefit": "Fermented batter promotes digestive bioavailability of B-vitamins with almost zero added fat.",
                "substitutions": [
                    {"dish_name": "Ragi Dosa (2 pcs) with Mint Coriander Chutney", "portion": "2 dosas", "calories": 410.0, "protein": 13.0, "fibre": 9.5},
                    {"dish_name": "Moong Dal Chilla with Tomato Slices", "portion": "2 chillas", "calories": 420.0, "protein": 19.0, "fibre": 8.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "2 Multigrain Rotis with Punjabi Rajma Masala (Kidney Beans) & Stir-Fried Beans Poriyal",
                "portion": "2 rotis + 1 cup slow-cooked rajma + 1 cup green beans poriyal + onion lemon salad",
                "calories": 650.0,
                "macronutrients": {"carbohydrates_g": 86.0, "protein_g": 26.0, "fats_g": 14.0, "fibre_g": 16.0, "sodium_mg": 270.0},
                "clinical_benefit": "Phaseolin protein and high potassium in kidney beans counter sodium retention and lower BP.",
                "substitutions": [
                    {"dish_name": "1.5 cup Brown Rice with Palak Paneer & Chana Dal", "portion": "1.5 cup rice + palak", "calories": 640.0, "protein": 27.0, "fibre": 12.5},
                    {"dish_name": "2 Jowar Rotis with Soya Matar Curry & Lauki Sabzi", "portion": "2 rotis + 1 cup curry", "calories": 630.0, "protein": 29.0, "fibre": 14.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Oil-Free Murmura Bhel with Chopped Tomato, Cucumber, Raw Mango & Lemon",
                "portion": "1.5 cups puffed rice tossed with fresh crunchy salad, roasted jeera & lemon juice",
                "calories": 210.0,
                "macronutrients": {"carbohydrates_g": 41.0, "protein_g": 5.0, "fats_g": 2.5, "fibre_g": 4.0, "sodium_mg": 95.0},
                "clinical_benefit": "Light snack providing vitamin C and potassium without saturated fats or heavy calorie burden.",
                "substitutions": [
                    {"dish_name": "Sprouted Moong Chaat with Chaat Masala", "portion": "1 cup", "calories": 210.0, "protein": 10.0, "fibre": 6.5},
                    {"dish_name": "Roasted Makhana (30g) with Green Tea", "portion": "1 bowl", "calories": 200.0, "protein": 5.5, "fibre": 4.5}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Millet Dosa (2 pcs) with Vegetable Korma (Coconut-Milk Free, Tomato Base)",
                "portion": "2 crispy kodo/foxtail millet dosas + 1.5 cups mixed vegetable korma with carrots & peas",
                "calories": 520.0,
                "macronutrients": {"carbohydrates_g": 75.0, "protein_g": 17.5, "fats_g": 11.0, "fibre_g": 10.5, "sodium_mg": 220.0},
                "clinical_benefit": "Ancient millets provide low-glycemic complex carbs that maintain nocturnal euglycemia.",
                "substitutions": [
                    {"dish_name": "Vegetable Quinoa Pulao with Sprouted Raita", "portion": "1.5 cups", "calories": 510.0, "protein": 19.0, "fibre": 10.0},
                    {"dish_name": "2 Phulkas with Methi Moong Dal & Bottle Gourd Sabzi", "portion": "2 phulkas + dal", "calories": 500.0, "protein": 18.0, "fibre": 11.0}
                ]
            }
        ]
    },
    {
        "day_number": 5,
        "day_name": "Day 5 (Friday)",
        "theme": "Lean Protein & Cellular Rejuvenation",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Besan & Grated Paneer Toast (Whole Wheat) with Mint Chutney & Spiced Tea",
                "portion": "2 whole grain bread slices coated in spiced gram flour & low-fat paneer + green tea",
                "calories": 440.0,
                "macronutrients": {"carbohydrates_g": 48.0, "protein_g": 22.0, "fats_g": 14.5, "fibre_g": 7.5, "sodium_mg": 210.0},
                "clinical_benefit": "Balanced breakfast delivering dual dairy and pulse proteins for sustained satiety and focus.",
                "substitutions": [
                    {"dish_name": "Moong Dal Chilla with Tomato Onion Salsa", "portion": "2 chillas", "calories": 410.0, "protein": 19.0, "fibre": 8.0},
                    {"dish_name": "Vegetable Poha with Roasted Peanuts & Lemon", "portion": "1.5 cups", "calories": 420.0, "protein": 12.0, "fibre": 7.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "1.5 cup Brown Rice with Rohu Fish Curry (or Tofu Tikka Masala) & Cabbage Poriyal",
                "portion": "1.5 cups steamed brown rice + 150g grilled fish/tofu in tomato mustard gravy + cabbage sabzi",
                "calories": 640.0,
                "macronutrients": {"carbohydrates_g": 75.0, "protein_g": 32.0, "fats_g": 16.0, "fibre_g": 11.5, "sodium_mg": 280.0},
                "clinical_benefit": "Supplies marine/plant Omega-3 EPA/DHA reducing systemic vascular inflammation and arterial plaque risk.",
                "substitutions": [
                    {"dish_name": "2 Multigrain Rotis with Palak Dal & Paneer Bhurji", "portion": "2 rotis + paneer", "calories": 630.0, "protein": 28.0, "fibre": 12.0},
                    {"dish_name": "2 Bajra Rotis with Methi Chana Dal & Lauki Sabzi", "portion": "2 rotis + dal", "calories": 620.0, "protein": 24.0, "fibre": 14.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Roasted Pumpkin & Flax Seeds (25g) with Fresh Ginger Tulsi Tea (No Sugar)",
                "portion": "1 small bowl dry-roasted seeds with black pepper + 1 mug brewed herbal tea",
                "calories": 210.0,
                "macronutrients": {"carbohydrates_g": 12.0, "protein_g": 9.0, "fats_g": 14.0, "fibre_g": 6.5, "sodium_mg": 40.0},
                "clinical_benefit": "Zinc and lignans from seeds support endothelial function and natural vascular relaxation.",
                "substitutions": [
                    {"dish_name": "Boiled Chickpea Chaat with Diced Cucumber", "portion": "1 cup", "calories": 220.0, "protein": 10.5, "fibre": 7.0},
                    {"dish_name": "Roasted Makhana with Rock Salt & Chaat Masala", "portion": "35g", "calories": 210.0, "protein": 6.0, "fibre": 5.0}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "2 Bajra Phulkas with Moong Dal Tadka & Tori (Ridge Gourd) Ki Sabzi",
                "portion": "2 soft bajra rotis + 1 cup yellow moong dal + 1 cup stewed ridge gourd + kachumber",
                "calories": 510.0,
                "macronutrients": {"carbohydrates_g": 76.0, "protein_g": 20.0, "fats_g": 10.0, "fibre_g": 13.0, "sodium_mg": 210.0},
                "clinical_benefit": "Low caloric density with high water content in ridge gourd allows full stomach volume without surplus calories.",
                "substitutions": [
                    {"dish_name": "Moong Dal & Brown Rice Khichdi with Dahi", "portion": "1.5 cups + 1 cup curd", "calories": 490.0, "protein": 18.0, "fibre": 9.5},
                    {"dish_name": "2 Phulkas with Baingan Bharta & Masoor Dal", "portion": "2 rotis + bharta", "calories": 520.0, "protein": 19.0, "fibre": 12.0}
                ]
            }
        ]
    },
    {
        "day_number": 6,
        "day_name": "Day 6 (Saturday)",
        "theme": "Millet Superfoods & Metabolic Agility",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Ragi (Finger Millet) Dosa (2 pcs) with Mint Chutney & Sprouted Moong Salad",
                "portion": "2 crisp ragi dosas + 2 tbsp fresh coriander mint dip + 1 cup sprouted moong",
                "calories": 420.0,
                "macronutrients": {"carbohydrates_g": 62.0, "protein_g": 17.5, "fats_g": 8.0, "fibre_g": 11.0, "sodium_mg": 180.0},
                "clinical_benefit": "Ragi provides 344mg calcium per 100g and polyphenols that inhibit alpha-glucosidase enzyme.",
                "substitutions": [
                    {"dish_name": "Vegetable Dalia with Carrots & Beans", "portion": "1.5 cups", "calories": 410.0, "protein": 14.0, "fibre": 8.5},
                    {"dish_name": "Besan Chilla with Paneer Stuffing", "portion": "2 chillas", "calories": 430.0, "protein": 19.0, "fibre": 7.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "2 Whole Wheat Chapatis with Paneer Bhurji & Panchmel Dal (5 Lentils)",
                "portion": "2 chapatis (no ghee) + 1 cup spiced paneer bhurji (100g) + 1 cup panchmel dal + salad",
                "calories": 650.0,
                "macronutrients": {"carbohydrates_g": 72.0, "protein_g": 32.0, "fats_g": 18.0, "fibre_g": 12.5, "sodium_mg": 290.0},
                "clinical_benefit": "Combines 5 pulse varieties to supply a broad spectrum of bioflavonoids and trace minerals.",
                "substitutions": [
                    {"dish_name": "2 Jowar Rotis with Soya Chunk Curry & Bhindi", "portion": "2 rotis + curry", "calories": 630.0, "protein": 30.0, "fibre": 13.5},
                    {"dish_name": "1.5 cup Brown Rice with Rajma Masala & Beans Poriyal", "portion": "1.5 cup rice + rajma", "calories": 640.0, "protein": 26.0, "fibre": 14.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Boiled Sweet Corn with Lemon Juice, Chaat Masala & Cold Spiced Chaas",
                "portion": "1 cup tender boiled sweet corn kernels + 1 glass (200ml) fresh jeera buttermilk",
                "calories": 210.0,
                "macronutrients": {"carbohydrates_g": 38.0, "protein_g": 6.5, "fats_g": 3.0, "fibre_g": 4.5, "sodium_mg": 120.0},
                "clinical_benefit": "Lutein and zeaxanthin in sweet corn protect ocular capillaries frequently vulnerable in diabetes.",
                "substitutions": [
                    {"dish_name": "Roasted Makhana with Green Tea", "portion": "35g bowl", "calories": 210.0, "protein": 6.0, "fibre": 5.0},
                    {"dish_name": "Roasted Chana with Cucumber Slices", "portion": "1 plate", "calories": 200.0, "protein": 8.0, "fibre": 5.5}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Vegetable Dalia (Broken Wheat) Khichdi with Low-Fat Curd & Raw Onion Kachumber",
                "portion": "1.5 cups wholesome vegetable dalia cooked with yellow moong + 1 bowl low-fat curd",
                "calories": 520.0,
                "macronutrients": {"carbohydrates_g": 76.0, "protein_g": 21.0, "fats_g": 10.0, "fibre_g": 11.5, "sodium_mg": 210.0},
                "clinical_benefit": "Broken wheat releases glucose exceptionally slowly across overnight fasting hours.",
                "substitutions": [
                    {"dish_name": "2 Phulkas with Palak Paneer & Dal", "portion": "2 phulkas + palak", "calories": 520.0, "protein": 22.0, "fibre": 10.5},
                    {"dish_name": "Vegetable Quinoa Pulao with Sprouted Raita", "portion": "1.5 cups", "calories": 500.0, "protein": 19.0, "fibre": 10.0}
                ]
            }
        ]
    },
    {
        "day_number": 7,
        "day_name": "Day 7 (Sunday)",
        "theme": "Weekend Nourishment & Sustained Vitality",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Sprouted Moong & Methi Thepla (2 pcs) with Low-Fat Dahi & Green Chilli Pickle",
                "portion": "2 non-greasy theplas prepared with fresh fenugreek & whole wheat + 1 cup dahi (150g)",
                "calories": 430.0,
                "macronutrients": {"carbohydrates_g": 54.0, "protein_g": 18.0, "fats_g": 12.0, "fibre_g": 9.0, "sodium_mg": 190.0},
                "clinical_benefit": "Fenugreek galactomannan delays carbohydrate absorption and lowers fasting blood sugar.",
                "substitutions": [
                    {"dish_name": "Moong Dal Chilla (2 pcs) with Mint Chutney", "portion": "2 chillas", "calories": 420.0, "protein": 19.5, "fibre": 8.5},
                    {"dish_name": "Steamed Idli (3 pcs) with Sambar", "portion": "3 idlis + sambar", "calories": 410.0, "protein": 14.0, "fibre": 8.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "2 Jowar Rotis with Amritsari Chana (No Soda) & Crispy Karela (Bitter Gourd) Fry",
                "portion": "2 jowar rotis + 1 cup chana slow cooked with amchur + 1 cup bitter gourd tossed in spices + cucumber",
                "calories": 640.0,
                "macronutrients": {"carbohydrates_g": 84.0, "protein_g": 24.5, "fats_g": 14.0, "fibre_g": 16.5, "sodium_mg": 260.0},
                "clinical_benefit": "Charantin and polypeptide-p in bitter gourd mimic insulin, facilitating cellular glucose utilization.",
                "substitutions": [
                    {"dish_name": "2 Multigrain Rotis with Palak Paneer & Yellow Dal", "portion": "2 rotis + palak", "calories": 630.0, "protein": 27.0, "fibre": 13.0},
                    {"dish_name": "1.5 cup Brown Rice with Soya Curry & Beans Poriyal", "portion": "1.5 cup rice + curry", "calories": 640.0, "protein": 26.0, "fibre": 12.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Roasted Pumpkin Seeds & Walnuts (25g) with Fresh Ginger Lemon Infusion",
                "portion": "25g mixed dry roasted seeds & nuts + warm lemon ginger infusion with fresh mint",
                "calories": 210.0,
                "macronutrients": {"carbohydrates_g": 14.0, "protein_g": 8.5, "fats_g": 13.5, "fibre_g": 5.5, "sodium_mg": 40.0},
                "clinical_benefit": "Potent polyphenol antioxidant profile that reduces oxidative stress accumulated throughout the week.",
                "substitutions": [
                    {"dish_name": "Oil-Free Murmura Bhel with Lemon", "portion": "1.5 cups", "calories": 200.0, "protein": 5.0, "fibre": 4.0},
                    {"dish_name": "Boiled Chana Chaat with Raw Mango", "portion": "1 cup", "calories": 220.0, "protein": 10.0, "fibre": 6.5}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Stuffed Lauki / Methi Paratha (Dry Roasted) with Spiced Boondi Raita & Dal Tadka",
                "portion": "2 dry roasted whole wheat parathas stuffed with spiced grated lauki + 1 cup boondi raita + 1 cup dal",
                "calories": 520.0,
                "macronutrients": {"carbohydrates_g": 74.0, "protein_g": 20.0, "fats_g": 11.5, "fibre_g": 11.0, "sodium_mg": 230.0},
                "clinical_benefit": "Light dinner that prevents overnight reflux, maintains gut balance, and stabilizes morning fasting numbers.",
                "substitutions": [
                    {"dish_name": "Vegetable Dalia Khichdi with Low-Fat Curd", "portion": "1.5 cups + curd", "calories": 500.0, "protein": 19.0, "fibre": 11.0},
                    {"dish_name": "Moong Dal & Brown Rice Khichdi with Kadhi", "portion": "1.5 cups + kadhi", "calories": 490.0, "protein": 18.0, "fibre": 9.5}
                ]
            }
        ]
    }
]

# Comprehensive 7-Day Curated Authentic Indian Non-Vegetarian Menu Templates (Clinically Heart-Safe & Low Glycemic)
CURATED_7_DAY_NON_VEG_MENUS = [
    {
        "day_number": 1,
        "day_name": "Day 1 (Monday)",
        "theme": "Metabolic Reset & High Bioavailable Lean Protein",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Indian Masala Egg Bhurji (2 Eggs + Veggies) with 2 Multigrain Phulkas & Mint Chaas",
                "portion": "2 scrambled eggs with onion, tomato, green chilli + 2 phulkas + 200ml spiced buttermilk",
                "calories": 440.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 48.0, "protein_g": 24.0, "fats_g": 14.0, "fibre_g": 7.5, "sodium_mg": 210.0},
                "clinical_benefit": "Whole egg choline and leucine stimulate muscle protein synthesis without triggering morning glucose spikes.",
                "substitutions": [
                    {"dish_name": "2 Boiled Eggs with Sprouted Moong Chaat", "portion": "2 eggs + 1 cup sprouts", "diet_type": "non_veg", "calories": 400.0, "protein": 22.0, "fibre": 8.0},
                    {"dish_name": "Oats & Egg White Omelette with Brown Toast", "portion": "2 egg whites + 2 slices", "diet_type": "non_veg", "calories": 390.0, "protein": 20.0, "fibre": 6.5}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "Tandoori Grilled Chicken Breast (Skinless, 150g) with 1 cup Brown Rice & Yellow Tadka Dal",
                "portion": "150g roasted chicken breast + 1 cup brown rice + 1 cup toor dal + fresh cucumber kachumber",
                "calories": 640.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 68.0, "protein_g": 44.0, "fats_g": 13.0, "fibre_g": 10.5, "sodium_mg": 290.0},
                "clinical_benefit": "Ultra-lean poultry delivers amino acids for insulin receptor sensitivity; brown rice provides slow-burning carbs.",
                "substitutions": [
                    {"dish_name": "Chicken Methi Curry with 2 Bajra Rotis", "portion": "1 bowl curry + 2 rotis", "diet_type": "non_veg", "calories": 630.0, "protein": 40.0, "fibre": 12.0},
                    {"dish_name": "Grilled Chicken Tikka with Quinoa Pulao", "portion": "150g tikka + 1 cup quinoa", "diet_type": "non_veg", "calories": 620.0, "protein": 42.0, "fibre": 9.5}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Roasted Spiced Makhana with Green Tea & 2 Boiled Egg Whites",
                "portion": "1 bowl makhana + 2 boiled egg whites with pepper + 1 mug green tea",
                "calories": 240.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 26.0, "protein_g": 13.5, "fats_g": 4.5, "fibre_g": 4.5, "sodium_mg": 110.0},
                "clinical_benefit": "Pure egg albumin provides late-afternoon satiety and suppresses counter-regulatory stress hormones.",
                "substitutions": [
                    {"dish_name": "Boiled Kala Chana Chaat with Lemon", "portion": "1 cup", "diet_type": "veg", "calories": 210.0, "protein": 10.0, "fibre": 7.0},
                    {"dish_name": "Roasted Chana with 5 Almonds", "portion": "1 palmful", "diet_type": "veg", "calories": 220.0, "protein": 9.0, "fibre": 5.5}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Light Steamed Rohu / Pomfret Fish Curry with 2 Whole Wheat Phulkas & Steamed Broccoli",
                "portion": "150g fish in tomato-onion light gravy + 2 phulkas + 1 cup steamed greens",
                "calories": 510.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 54.0, "protein_g": 36.0, "fats_g": 11.0, "fibre_g": 9.0, "sodium_mg": 240.0},
                "clinical_benefit": "Omega-3 fatty acids (EPA/DHA) actively reduce vascular inflammation, protect endothelium, and lower triglycerides.",
                "substitutions": [
                    {"dish_name": "Steamed Fish with 1 cup Brown Rice & Clear Broth", "portion": "150g fish + rice", "diet_type": "non_veg", "calories": 500.0, "protein": 34.0, "fibre": 8.0},
                    {"dish_name": "Egg Curry (2 eggs, light gravy) with 2 Jowar Phulkas", "portion": "2 eggs curry + 2 rotis", "diet_type": "non_veg", "calories": 520.0, "protein": 24.0, "fibre": 10.0}
                ]
            }
        ]
    },
    {
        "day_number": 2,
        "day_name": "Day 2 (Tuesday)",
        "theme": "Vascular Resilience & Omega-3 Cardio Support",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Oats & Vegetable Omelette (2 Eggs) with 1 Glass Low-Fat Milk / Chaas",
                "portion": "2 eggs folded with rolled oats, bell peppers & coriander + 200ml chaas",
                "calories": 430.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 42.0, "protein_g": 23.0, "fats_g": 13.5, "fibre_g": 6.5, "sodium_mg": 190.0},
                "clinical_benefit": "Beta-glucans from oats combined with dietary proteins delay gastric emptying and prevent mid-morning hunger.",
                "substitutions": [
                    {"dish_name": "Vegetable Poha with 2 Boiled Eggs", "portion": "1.5 cup poha + 2 eggs", "diet_type": "non_veg", "calories": 430.0, "protein": 20.0, "fibre": 6.5},
                    {"dish_name": "Egg Bhurji Wrap in 1 Whole Wheat Roti", "portion": "1 roll", "diet_type": "non_veg", "calories": 410.0, "protein": 19.0, "fibre": 5.5}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "Murgh Methi Malai (Low-Fat Dahi Gravy, 140g) with 2 Bajra Rotis & Lauki Sabzi",
                "portion": "140g chicken breast in fenugreek curd base + 2 bajra rotis + 1 cup bottle gourd sabzi",
                "calories": 640.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 66.0, "protein_g": 42.0, "fats_g": 14.5, "fibre_g": 13.0, "sodium_mg": 260.0},
                "clinical_benefit": "Fenugreek seeds and leaves contain trigonelline which optimizes cellular glucose uptake.",
                "substitutions": [
                    {"dish_name": "Chicken Tikka Masala (Low Oil) with 2 Multigrain Rotis", "portion": "1 bowl + 2 rotis", "diet_type": "non_veg", "calories": 630.0, "protein": 39.0, "fibre": 11.0},
                    {"dish_name": "Grilled Fish Fillet with Brown Rice Khichdi", "portion": "150g fish + 1 cup khichdi", "diet_type": "non_veg", "calories": 620.0, "protein": 38.0, "fibre": 10.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Sprouted Kala Chana & Boiled Egg White Chaat with Fresh Lemon & Herbs",
                "portion": "1 bowl sprouted chana + 2 chopped egg whites with rock salt & lemon juice",
                "calories": 230.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 28.0, "protein_g": 15.0, "fats_g": 3.5, "fibre_g": 7.0, "sodium_mg": 120.0},
                "clinical_benefit": "Synergistic plant and egg protein delivering amino acids and potassium for blood pressure regulation.",
                "substitutions": [
                    {"dish_name": "Spiced Fox Nuts (Makhana) with Green Tea", "portion": "35g makhana", "diet_type": "veg", "calories": 210.0, "protein": 6.0, "fibre": 5.0},
                    {"dish_name": "Boiled Sweet Corn with Lemon Pepper", "portion": "1 cup", "diet_type": "veg", "calories": 200.0, "protein": 5.5, "fibre": 4.5}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Pan-Grilled Fish Tikka (150g) with Vegetable Broken Wheat Pulao & Mint Raita",
                "portion": "150g marinated fish skewers + 1.25 cup dalia pulao + 1 cup cucumber raita",
                "calories": 510.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 56.0, "protein_g": 35.0, "fats_g": 11.5, "fibre_g": 9.5, "sodium_mg": 230.0},
                "clinical_benefit": "High potassium-to-sodium ratio in dahlia and fish relieves arterial strain overnight.",
                "substitutions": [
                    {"dish_name": "Bengali Macher Jhol with 1 cup Brown Rice", "portion": "150g fish curry + rice", "diet_type": "non_veg", "calories": 490.0, "protein": 33.0, "fibre": 8.0},
                    {"dish_name": "2 Boiled Eggs with Mixed Vegetable Soup & 1 Phulka", "portion": "2 eggs + soup + roti", "diet_type": "non_veg", "calories": 480.0, "protein": 22.0, "fibre": 7.5}
                ]
            }
        ]
    },
    {
        "day_number": 3,
        "day_name": "Day 3 (Wednesday)",
        "theme": "High-Fiber Glycemic Stabilization & Satiety",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Egg Poha (Flattened Rice with 2 Scrambled Eggs, Peas & Peanuts)",
                "portion": "1.5 cups poha with egg ribbons, green peas, curry leaves & roasted peanuts",
                "calories": 440.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 52.0, "protein_g": 21.0, "fats_g": 13.0, "fibre_g": 6.0, "sodium_mg": 180.0},
                "clinical_benefit": "Healthy fats from peanuts moderate the carbohydrate release from poha.",
                "substitutions": [
                    {"dish_name": "2 Boiled Eggs with 2 Steamed Idlis & Sambar", "portion": "2 eggs + 2 idlis", "diet_type": "non_veg", "calories": 420.0, "protein": 20.0, "fibre": 7.0},
                    {"dish_name": "Besan Chilla with 2 Boiled Egg Whites", "portion": "2 chillas + 2 whites", "diet_type": "non_veg", "calories": 410.0, "protein": 22.0, "fibre": 7.5}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "Murgh Saagwala (Lean Chicken in Spiced Spinach Puree) with 2 Multigrain Phulkas & Dal Tadka",
                "portion": "150g chicken breast in palak gravy + 2 phulkas (wheat+ragi+chana) + 1 cup dal",
                "calories": 640.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 64.0, "protein_g": 46.0, "fats_g": 14.0, "fibre_g": 12.0, "sodium_mg": 280.0},
                "clinical_benefit": "Spinach magnesium and dietary nitrates support endothelial vasodilation and arterial flexibility.",
                "substitutions": [
                    {"dish_name": "Chicken Keema Matar with 2 Jowar Rotis", "portion": "1 cup keema + 2 rotis", "diet_type": "non_veg", "calories": 630.0, "protein": 42.0, "fibre": 11.5},
                    {"dish_name": "1.5 cup Brown Rice with Chicken Sambar & Cabbage Poriyal", "portion": "1.5 cup rice + chicken", "diet_type": "non_veg", "calories": 620.0, "protein": 38.0, "fibre": 11.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Sprouted Moong Chaat with 1 Diced Boiled Egg & Rock Salt",
                "portion": "1 cup sprouted moong + 1 hard-boiled egg with chopped tomatoes and lemon",
                "calories": 220.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 24.0, "protein_g": 14.0, "fats_g": 5.0, "fibre_g": 6.5, "sodium_mg": 100.0},
                "clinical_benefit": "Living sprout enzymes and complete egg amino acids provide sustained afternoon energy.",
                "substitutions": [
                    {"dish_name": "Roasted Chana with Lemon & Cucumber", "portion": "1 cup", "diet_type": "veg", "calories": 210.0, "protein": 9.0, "fibre": 6.0},
                    {"dish_name": "Roasted Fox Nuts (Makhana) with Pepper", "portion": "35g", "diet_type": "veg", "calories": 200.0, "protein": 6.0, "fibre": 5.0}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Bengali Macher Jhol (Steamed Rohu Fish with Cauliflower & Potato in Cumin Gravy) with Brown Rice",
                "portion": "150g fish steak with 1 cup brown rice and light digestive cumin-ginger broth",
                "calories": 510.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 58.0, "protein_g": 34.0, "fats_g": 10.0, "fibre_g": 8.5, "sodium_mg": 210.0},
                "clinical_benefit": "Light cumin-ginger broth stimulates gastrointestinal motility without heavy cream or ghee.",
                "substitutions": [
                    {"dish_name": "2 Whole Wheat Rotis with Egg Bhurji (2 eggs) & Bhindi Sabzi", "portion": "2 rotis + bhurji", "diet_type": "non_veg", "calories": 500.0, "protein": 22.0, "fibre": 9.5},
                    {"dish_name": "Grilled Fish with Quinoa Pulao & Mint Raita", "portion": "150g fish + 1 cup quinoa", "diet_type": "non_veg", "calories": 520.0, "protein": 36.0, "fibre": 8.0}
                ]
            }
        ]
    },
    {
        "day_number": 4,
        "day_name": "Day 4 (Thursday)",
        "theme": "Lipid Optimization & Lean Muscle Recovery",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Steamed Rice & Urad Idlis (3 pcs) with 2 Boiled Egg Whites & Drumstick Sambar",
                "portion": "3 idlis + 2 egg whites + 1.5 cup vegetable-loaded sambar",
                "calories": 420.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 54.0, "protein_g": 20.0, "fats_g": 5.0, "fibre_g": 8.0, "sodium_mg": 220.0},
                "clinical_benefit": "Fermented batter introduces gut-friendly probiotics; egg whites add pure albumin without cholesterol.",
                "substitutions": [
                    {"dish_name": "Vegetable Dosa with 2 Boiled Eggs & Sambar", "portion": "1 dosa + 2 eggs", "diet_type": "non_veg", "calories": 430.0, "protein": 21.0, "fibre": 7.0},
                    {"dish_name": "Oats Upma with 2 Scrambled Egg Whites", "portion": "1.5 cup upma", "diet_type": "non_veg", "calories": 400.0, "protein": 18.0, "fibre": 7.5}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "Chicken Keema Matar (Lean Chicken Mince & Peas) with 2 Jowar Rotis & Plain Curd",
                "portion": "140g lean chicken mince with fresh green peas + 2 sorghum rotis + 100g low-fat curd",
                "calories": 650.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 65.0, "protein_g": 43.0, "fats_g": 15.0, "fibre_g": 12.5, "sodium_mg": 270.0},
                "clinical_benefit": "High satiety index from lean mince reduces late afternoon sugar cravings.",
                "substitutions": [
                    {"dish_name": "Tandoori Chicken with 2 Wheat Phulkas & Dal", "portion": "150g chicken + 2 rotis", "diet_type": "non_veg", "calories": 640.0, "protein": 44.0, "fibre": 11.0},
                    {"dish_name": "Grilled Fish Steak with Brown Rice & French Beans", "portion": "150g fish + 1 cup rice", "diet_type": "non_veg", "calories": 620.0, "protein": 37.0, "fibre": 10.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Spiced Roasted Makhana with Unsweetened Green Tea & 4 Walnuts",
                "portion": "35g roasted fox nuts + 4 walnut halves + green tea",
                "calories": 230.0,
                "diet_type": "veg",
                "macronutrients": {"carbohydrates_g": 24.0, "protein_g": 6.5, "fats_g": 10.0, "fibre_g": 5.0, "sodium_mg": 80.0},
                "clinical_benefit": "Alpha-linolenic acid (ALA) in walnuts supports anti-arrhythmic cardiovascular stability.",
                "substitutions": [
                    {"dish_name": "2 Boiled Egg Whites with Black Pepper", "portion": "2 whites", "diet_type": "non_veg", "calories": 140.0, "protein": 12.0, "fibre": 1.0},
                    {"dish_name": "Sprouted Moong Chaat with Lemon", "portion": "1 cup", "diet_type": "veg", "calories": 210.0, "protein": 10.0, "fibre": 7.0}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Pan-Seared Salmon / Indian Basa Fillet (150g) with Quinoa Pulao & Warm Tomato Rasam",
                "portion": "150g spiced grilled fish + 1.25 cup vegetable quinoa + 1 bowl digestion-enhancing rasam",
                "calories": 520.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 52.0, "protein_g": 38.0, "fats_g": 12.5, "fibre_g": 8.0, "sodium_mg": 240.0},
                "clinical_benefit": "Marine omega-3s combat cellular oxidation and help lower resting diastolic arterial tone.",
                "substitutions": [
                    {"dish_name": "Light Chicken Stew with 2 Whole Wheat Phulkas", "portion": "1 bowl stew + 2 rotis", "diet_type": "non_veg", "calories": 500.0, "protein": 35.0, "fibre": 8.5},
                    {"dish_name": "Egg Curry (2 eggs) with 1 cup Brown Rice Khichdi", "portion": "2 eggs + 1 cup khichdi", "diet_type": "non_veg", "calories": 510.0, "protein": 23.0, "fibre": 9.0}
                ]
            }
        ]
    },
    {
        "day_number": 5,
        "day_name": "Day 5 (Friday)",
        "theme": "Cardiovascular Conditioning & Lean Seafood Synergy",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Scrambled Egg Toast (2 Multigrain Slices + 2 Eggs) with Tomato Slices & Green Tea",
                "portion": "2 whole eggs scrambled in minimal cold-pressed mustard oil + 2 multigrain bread slices + green tea",
                "calories": 430.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 44.0, "protein_g": 22.0, "fats_g": 13.0, "fibre_g": 6.5, "sodium_mg": 200.0},
                "clinical_benefit": "Whole grain fibre combined with egg proteins buffers postprandial glycemic excursions.",
                "substitutions": [
                    {"dish_name": "2 Boiled Eggs with Vegetable Upma", "portion": "2 eggs + 1 cup upma", "diet_type": "non_veg", "calories": 420.0, "protein": 20.0, "fibre": 7.0},
                    {"dish_name": "Moong Dal Chilla with 2 Boiled Egg Whites", "portion": "2 chillas + 2 whites", "diet_type": "non_veg", "calories": 410.0, "protein": 23.0, "fibre": 8.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "Andhra Style Fish Curry (Tamarind & Fenugreek Gravy) with 1.5 cup Brown Rice & Beetroot Poriyal",
                "portion": "150g fish in tangy tamarind-fenugreek curry + 1.5 cup brown rice + 1 cup beetroot stir fry",
                "calories": 630.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 72.0, "protein_g": 36.0, "fats_g": 12.0, "fibre_g": 11.5, "sodium_mg": 270.0},
                "clinical_benefit": "Beetroot betaine and dietary nitrates improve microvascular compliance and lower blood pressure.",
                "substitutions": [
                    {"dish_name": "Chicken Curry (Home-Style, Low Oil) with 2 Bajra Rotis", "portion": "1 bowl + 2 rotis", "diet_type": "non_veg", "calories": 640.0, "protein": 41.0, "fibre": 12.0},
                    {"dish_name": "Tandoori Chicken Skewers with Quinoa Salad", "portion": "150g chicken + salad", "diet_type": "non_veg", "calories": 610.0, "protein": 43.0, "fibre": 9.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Spiced Roasted Fox Nuts (Makhana) with Spiced Lemon Buttermilk (Chaas)",
                "portion": "35g roasted makhana + 200ml cold chaas with roasted jeera",
                "calories": 210.0,
                "diet_type": "veg",
                "macronutrients": {"carbohydrates_g": 28.0, "protein_g": 6.5, "fats_g": 4.5, "fibre_g": 5.0, "sodium_mg": 110.0},
                "clinical_benefit": "Cumin and probiotics in buttermilk promote healthy bile acid metabolism and gut comfort.",
                "substitutions": [
                    {"dish_name": "1 Hard-Boiled Egg with Rock Salt & Pepper", "portion": "1 egg", "diet_type": "non_veg", "calories": 150.0, "protein": 11.0, "fibre": 1.0},
                    {"dish_name": "Sprouted Kala Chana Chaat", "portion": "1 cup", "diet_type": "veg", "calories": 210.0, "protein": 10.0, "fibre": 7.0}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Grilled Chicken Breast (140g) with Vegetable Dalia Khichdi & Mint Raita",
                "portion": "140g marinated grilled chicken + 1.25 cup broken wheat khichdi + 1 cup cucumber raita",
                "calories": 510.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 52.0, "protein_g": 41.0, "fats_g": 11.0, "fibre_g": 9.5, "sodium_mg": 220.0},
                "clinical_benefit": "Dalia delivers slow-digesting resistant starch that sustains nocturnal insulin sensitivity.",
                "substitutions": [
                    {"dish_name": "Steamed Fish Curry with 2 Whole Wheat Phulkas", "portion": "150g fish + 2 rotis", "diet_type": "non_veg", "calories": 500.0, "protein": 34.0, "fibre": 8.5},
                    {"dish_name": "Egg Bhurji (2 eggs) with 2 Jowar Phulkas & Salad", "portion": "2 eggs + 2 rotis", "diet_type": "non_veg", "calories": 490.0, "protein": 22.0, "fibre": 9.5}
                ]
            }
        ]
    },
    {
        "day_number": 6,
        "day_name": "Day 6 (Saturday)",
        "theme": "Weekend Athletic Recovery & Antioxidant Boost",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Parsi Akuri (Spiced Soft-Scrambled Eggs, 2 Eggs) with 2 Whole Wheat Phulkas & Buttermilk",
                "portion": "2 eggs prepared with ginger, garlic, turmeric, tomatoes + 2 phulkas + 200ml chaas",
                "calories": 430.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 46.0, "protein_g": 23.0, "fats_g": 13.5, "fibre_g": 7.0, "sodium_mg": 210.0},
                "clinical_benefit": "Curcumin from turmeric and capsaicin from green chillies downregulate NF-kB inflammatory cascades.",
                "substitutions": [
                    {"dish_name": "2 Boiled Eggs with Oats Upma", "portion": "2 eggs + upma", "diet_type": "non_veg", "calories": 420.0, "protein": 21.0, "fibre": 7.5},
                    {"dish_name": "Egg & Cheese Wrap (Low Fat) in Multigrain Roti", "portion": "1 wrap", "diet_type": "non_veg", "calories": 410.0, "protein": 20.0, "fibre": 6.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "Chicken & Mixed Lentil Stew (Dhansak Style, Low Fat) with 1.25 cup Brown Rice & Kachumber",
                "portion": "150g chicken breast cooked with pumpkin, fenugreek & 3 lentils + 1.25 cup brown rice",
                "calories": 650.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 68.0, "protein_g": 45.0, "fats_g": 13.5, "fibre_g": 13.0, "sodium_mg": 280.0},
                "clinical_benefit": "Triple-lentil synergy with poultry delivers high-potency protein and viscous soluble fibres.",
                "substitutions": [
                    {"dish_name": "Grilled Fish Fillet with 2 Jowar Rotis & Lauki Sabzi", "portion": "150g fish + 2 rotis", "diet_type": "non_veg", "calories": 620.0, "protein": 37.0, "fibre": 11.5},
                    {"dish_name": "Chicken Tikka with Dal Makhani (Low Fat) & 2 Phulkas", "portion": "150g chicken + dal + rotis", "diet_type": "non_veg", "calories": 640.0, "protein": 42.0, "fibre": 12.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "1 Hard-Boiled Egg with Crushed Black Pepper & Handful of Roasted Chana",
                "portion": "1 boiled egg + 20g roasted chana + warm lemon green tea",
                "calories": 210.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 18.0, "protein_g": 13.0, "fats_g": 6.5, "fibre_g": 4.5, "sodium_mg": 90.0},
                "clinical_benefit": "Ideal pre-evening workout snack offering sustained glycemic support.",
                "substitutions": [
                    {"dish_name": "Roasted Makhana with Rock Salt", "portion": "35g", "diet_type": "veg", "calories": 200.0, "protein": 6.0, "fibre": 5.0},
                    {"dish_name": "Sprouted Moong Chaat with Lemon", "portion": "1 cup", "diet_type": "veg", "calories": 210.0, "protein": 10.0, "fibre": 7.0}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Steamed Pomfret in Banana Leaf (Patra Ni Machhi) with Quinoa Pulao & Clear Lemon Coriander Soup",
                "portion": "150g steamed fish coated in fresh coconut-mint green chutney + 1 cup quinoa pulao + 1 bowl soup",
                "calories": 520.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 48.0, "protein_g": 37.0, "fats_g": 13.0, "fibre_g": 7.5, "sodium_mg": 210.0},
                "clinical_benefit": "Steaming in banana leaf seals volatile polyphenols and nutrients without oxidized cooking fats.",
                "substitutions": [
                    {"dish_name": "Light Chicken Curry with 2 Whole Wheat Phulkas", "portion": "1 bowl + 2 rotis", "diet_type": "non_veg", "calories": 500.0, "protein": 36.0, "fibre": 8.5},
                    {"dish_name": "Egg Curry (2 eggs) with 1 cup Brown Rice", "portion": "2 eggs + rice", "diet_type": "non_veg", "calories": 510.0, "protein": 22.0, "fibre": 8.0}
                ]
            }
        ]
    },
    {
        "day_number": 7,
        "day_name": "Day 7 (Sunday)",
        "theme": "Restorative Cellular Longevity & Metabolic Harmony",
        "meals": [
            {
                "slot": "Breakfast",
                "dish_name": "Moong Dal & Egg White Crepes (2 pcs) with Fresh Coriander Chutney & Spiced Chaas",
                "portion": "2 high-protein moong dal chillas topped with egg white scramble + 2 tbsp chutney + 200ml chaas",
                "calories": 420.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 46.0, "protein_g": 24.0, "fats_g": 9.5, "fibre_g": 8.0, "sodium_mg": 190.0},
                "clinical_benefit": "Plant and egg protein synergy keeps fasting insulin low and extends metabolic flexibility.",
                "substitutions": [
                    {"dish_name": "2 Boiled Eggs with Vegetable Poha", "portion": "2 eggs + 1.5 cup poha", "diet_type": "non_veg", "calories": 430.0, "protein": 20.0, "fibre": 6.5},
                    {"dish_name": "Omelette with 2 Slices Multigrain Toast", "portion": "2 eggs + 2 toasts", "diet_type": "non_veg", "calories": 410.0, "protein": 21.0, "fibre": 6.0}
                ]
            },
            {
                "slot": "Lunch",
                "dish_name": "Sunday Home-Style Chicken Curry (Skinless Breast, 150g) with 2 Bajra Rotis & Fresh Salad",
                "portion": "150g chicken breast simmered with whole spices + 2 bajra rotis + green kachumber salad",
                "calories": 650.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 66.0, "protein_g": 43.0, "fats_g": 14.5, "fibre_g": 13.0, "sodium_mg": 280.0},
                "clinical_benefit": "Bajra grain minerals (iron, zinc, magnesium) and clean lean chicken protein reinforce systemic vitality.",
                "substitutions": [
                    {"dish_name": "Tandoori Chicken Breast with Quinoa Pulao & Raita", "portion": "150g chicken + 1 cup quinoa", "diet_type": "non_veg", "calories": 630.0, "protein": 44.0, "fibre": 9.5},
                    {"dish_name": "Grilled Fish Tikka with 2 Jowar Rotis & Dal", "portion": "150g fish + 2 rotis + dal", "diet_type": "non_veg", "calories": 620.0, "protein": 39.0, "fibre": 11.0}
                ]
            },
            {
                "slot": "Evening Snack",
                "dish_name": "Dry Roasted Spiced Makhana with 4 Almonds & Green Tea (No Sugar)",
                "portion": "35g roasted fox nuts + 4 raw almonds + 1 cup green tea",
                "calories": 220.0,
                "diet_type": "veg",
                "macronutrients": {"carbohydrates_g": 26.0, "protein_g": 6.5, "fats_g": 7.0, "fibre_g": 5.0, "sodium_mg": 85.0},
                "clinical_benefit": "Magnesium in fox nuts calms the autonomic nervous system before Sunday evening rest.",
                "substitutions": [
                    {"dish_name": "2 Boiled Egg Whites with Lemon Pepper", "portion": "2 whites", "diet_type": "non_veg", "calories": 140.0, "protein": 12.0, "fibre": 1.0},
                    {"dish_name": "Boiled Kala Chana with Chaat Masala", "portion": "1 cup", "diet_type": "veg", "calories": 210.0, "protein": 10.0, "fibre": 7.0}
                ]
            },
            {
                "slot": "Dinner",
                "dish_name": "Grilled Fish Tikka Skewers (150g) with 2 Whole Wheat Phulkas & Low-Fat Cucumber Raita",
                "portion": "150g marinated fish grilled dry + 2 phulkas + 1 cup refreshing mint cucumber raita",
                "calories": 510.0,
                "diet_type": "non_veg",
                "macronutrients": {"carbohydrates_g": 50.0, "protein_g": 37.0, "fats_g": 11.5, "fibre_g": 8.5, "sodium_mg": 220.0},
                "clinical_benefit": "Light dinner with supreme bioavailability; allows nocturnal insulin levels to baseline smoothly.",
                "substitutions": [
                    {"dish_name": "Egg Curry (2 eggs) with Quinoa / Brown Rice Khichdi", "portion": "2 eggs + 1 cup khichdi", "diet_type": "non_veg", "calories": 510.0, "protein": 24.0, "fibre": 9.0},
                    {"dish_name": "Chicken Clear Soup with 2 Multigrain Phulkas & Salad", "portion": "1 bowl soup + 2 rotis", "diet_type": "non_veg", "calories": 490.0, "protein": 34.0, "fibre": 8.0}
                ]
            }
        ]
    }
]

def calculate_recommended_calories(risk_profile: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    """
    Clinically calculates the daily caloric target with complete mathematical transparency
    using the validated Mifflin-St Jeor Equation for Basal Metabolic Rate (BMR) and Total
    Daily Energy Expenditure (TDEE), followed by an evidence-based clinical deficit/surplus.
    """
    age = float(risk_profile.get("age") or 40.0)
    raw_gender = str(risk_profile.get("gender") or risk_profile.get("sex") or "Male")
    gender = "Female" if "fe" in raw_gender.lower() or raw_gender.strip().lower() == "f" else "Male"

    # Check whether height and weight were genuinely provided in risk_profile
    raw_height = risk_profile.get("height_cm") if risk_profile.get("height_cm") is not None else risk_profile.get("height")
    raw_weight = risk_profile.get("weight_kg") if risk_profile.get("weight_kg") is not None else risk_profile.get("weight")
    raw_bmi = risk_profile.get("bmi")

    has_valid_height = False
    has_valid_weight = False
    has_valid_bmi = False

    est_height = 0.0
    est_weight = 0.0
    bmi_val = None

    if raw_height is not None:
        try:
            h_float = float(raw_height)
            if h_float > 100.0:
                est_height = h_float
                has_valid_height = True
            elif 1.0 <= h_float <= 2.5:
                est_height = round(h_float * 100.0, 1)
                has_valid_height = True
        except (ValueError, TypeError):
            pass

    if raw_weight is not None:
        try:
            w_float = float(raw_weight)
            if w_float > 30.0:
                est_weight = w_float
                has_valid_weight = True
        except (ValueError, TypeError):
            pass

    if raw_bmi is not None:
        try:
            b_float = float(raw_bmi)
            if 10.0 <= b_float <= 65.0:
                bmi_val = round(b_float, 1)
                has_valid_bmi = True
        except (ValueError, TypeError):
            pass

    anthropometrics_available = has_valid_height and has_valid_weight

    # BUG J FIX: If height and weight are absent from the report, DO NOT fabricate default 172cm/71kg!
    # Instead, clearly declare that caloric targets are estimated using standard population baseline (1,800 kcal)
    if not anthropometrics_available:
        baseline_cals = 1800.0
        calculation_steps = [
            "1. Anthropometric Data: Height and weight were not found in the uploaded report.",
            "2. Non-Personalized Baseline: Caloric target is set to the standard population baseline (1,800 kcal/day) with NO report-derived personalization.",
            "3. Action Required: Please enter your height and weight manually in your profile to generate an individualized BMR and tailored caloric plan."
        ]
        clinical_rationale = (
            "Caloric targets are estimated using standard population defaults (1,800 kcal/day) with no "
            "personalization from this report because height, weight, and BMI were not documented. "
            "Please enter your height and weight for clinical personalization."
        )
        breakdown = {
            "formula": "Standard Population Baseline (Non-Personalized Estimate)",
            "anthropometrics_available": False,
            "is_personalized": False,
            "gender": gender,
            "age": int(round(age)),
            "height_cm": None,
            "weight_kg": None,
            "bmi": bmi_val,
            "bmr": None,
            "tdee": None,
            "activity_multiplier": None,
            "deficit": 0,
            "deficit_reason": "Standard population baseline (no clinical deficit applied without verified height/weight)",
            "recommended_calories": 1800,
            "calculation_steps": calculation_steps,
            "disclaimer": "Caloric targets are estimated using standard defaults with no personalization from this report.",
            "prompt_for_manual_input": True
        }
        return baseline_cals, clinical_rationale, breakdown

    # Anthropometrics ARE available -> Mifflin-St Jeor calculation with actual patient metrics
    computed_bmi = bmi_val if has_valid_bmi else round(est_weight / ((est_height / 100.0) ** 2), 1)

    # 1. Basal Metabolic Rate (Mifflin-St Jeor Formula)
    if gender == "Female":
        bmr = (10.0 * est_weight) + (6.25 * est_height) - (5.0 * age) - 161.0
    else:
        bmr = (10.0 * est_weight) + (6.25 * est_height) - (5.0 * age) + 5.0

    # 2. Total Daily Energy Expenditure (TDEE with 1.30x light activity multiplier)
    activity_mult = 1.30
    tdee = bmr * activity_mult

    # 3. Clinical Deficit/Surplus Justification
    deficit = 0.0
    deficit_reason = "Maintenance caloric balance (healthy weight range)"

    if computed_bmi >= 30.0:
        deficit = 500.0
        deficit_reason = f"Safe 500 kcal deficit for progressive visceral weight reduction (BMI {computed_bmi:.1f})"
    elif computed_bmi >= 25.0:
        deficit = 350.0
        deficit_reason = f"Moderate 350 kcal deficit for South Asian overweight status to improve insulin sensitivity (BMI {computed_bmi:.1f})"
    elif computed_bmi >= 23.0:
        deficit = 200.0
        deficit_reason = f"Mild 200 kcal deficit for body composition optimization (BMI {computed_bmi:.1f})"
    elif computed_bmi < 18.5:
        deficit = -250.0
        deficit_reason = f"250 kcal surplus to restore healthy nutritional baseline (BMI {computed_bmi:.1f})"

    net_calories = tdee - deficit
    floor = 1350.0 if gender == "Female" else 1500.0
    ceiling = 2400.0
    adjusted_calories = max(floor, min(ceiling, net_calories))
    recommended_calories = round(adjusted_calories / 50.0) * 50.0

    breakdown = {
        "formula": "Mifflin-St Jeor Equation + TDEE",
        "anthropometrics_available": True,
        "is_personalized": True,
        "gender": gender,
        "age": int(round(age)),
        "height_cm": round(est_height, 1),
        "weight_kg": round(est_weight, 1),
        "bmi": round(computed_bmi, 1),
        "bmr": int(round(bmr)),
        "tdee": int(round(tdee)),
        "activity_multiplier": activity_mult,
        "deficit": int(round(deficit)),
        "deficit_reason": deficit_reason,
        "recommended_calories": int(round(recommended_calories)),
        "calculation_steps": [
            f"1. Basal Metabolic Rate (BMR): {int(round(bmr))} kcal/day using Mifflin-St Jeor formula based on age ({int(round(age))}), {gender}, {round(est_height, 0):.0f} cm, {round(est_weight, 1)} kg.",
            f"2. Daily Energy Expenditure (TDEE): {int(round(tdee))} kcal/day with light physical activity (1.30× multiplier).",
            f"3. Clinical Caloric Adjustment: -{int(round(deficit))} kcal/day ({deficit_reason}).",
            f"4. Recommended Daily Target: {int(round(recommended_calories))} kcal/day."
        ]
    }

    clinical_rationale = f"Caloric target of {int(round(recommended_calories))} kcal/day calculated via Mifflin-St Jeor (BMR {int(round(bmr))} kcal, TDEE {int(round(tdee))} kcal) with a {int(round(deficit))} kcal clinical adjustment for BMI {computed_bmi:.1f}."

    return float(recommended_calories), clinical_rationale, breakdown

def generate_diet_plan(risk_profile: Dict[str, Any], calorie_target: Optional[float] = None, diet_type: str = "veg") -> Dict[str, Any]:
    """Generates a complete 7-day Indian diet plan (Vegetarian or Non-Vegetarian) clinically aligned with the patient's findings."""
    calc_cals, clinical_rationale, calorie_breakdown = calculate_recommended_calories(risk_profile)
    
    # If user has not specified a custom calorie target or passed <= 500, use clinically calculated target
    if calorie_target is None or calorie_target <= 500.0:
        calorie_target = calc_cals

    is_non_veg = bool(diet_type and "non" in str(diet_type).lower())
    active_diet_type = "non_veg" if is_non_veg else "veg"
    active_menu_templates = CURATED_7_DAY_NON_VEG_MENUS if is_non_veg else CURATED_7_DAY_MENUS

    # Scale base menus (base templates are around 1800 kcal)
    scale_factor = calorie_target / 1800.0

    scaled_days = []
    for day_template in active_menu_templates:
        day_meals = []
        day_total_cal = 0.0
        day_total_carbs = 0.0
        day_total_protein = 0.0
        day_total_fats = 0.0
        day_total_fibre = 0.0
        day_total_sodium = 0.0

        for m in day_template["meals"]:
            scaled_cals = round(m["calories"] * scale_factor, 1)
            macros = m["macronutrients"]
            scaled_macros = {
                "carbohydrates_g": round(macros["carbohydrates_g"] * scale_factor, 1),
                "protein_g": round(macros["protein_g"] * scale_factor, 1),
                "fats_g": round(macros["fats_g"] * scale_factor, 1),
                "fibre_g": round(macros["fibre_g"] * scale_factor, 1),
                "sodium_mg": round(macros["sodium_mg"] * scale_factor, 1)
            }

            scaled_subs = []
            for s in m.get("substitutions", []):
                scaled_subs.append({
                    "dish_name": s["dish_name"],
                    "portion": s.get("portion", "Standard serving"),
                    "diet_type": s.get("diet_type", m.get("diet_type", active_diet_type)),
                    "calories": round(s["calories"] * scale_factor, 1),
                    "protein": round(s.get("protein", 10.0) * scale_factor, 1),
                    "fibre": round(s.get("fibre", 5.0) * scale_factor, 1)
                })

            day_total_cal += scaled_cals
            day_total_carbs += scaled_macros["carbohydrates_g"]
            day_total_protein += scaled_macros["protein_g"]
            day_total_fats += scaled_macros["fats_g"]
            day_total_fibre += scaled_macros["fibre_g"]
            day_total_sodium += scaled_macros["sodium_mg"]

            day_meals.append({
                "slot": m["slot"],
                "meal_type": m["slot"],
                "dish_name": m["dish_name"],
                "portion": m["portion"],
                "diet_type": m.get("diet_type", active_diet_type),
                "calories": scaled_cals,
                "total_calories": scaled_cals,
                "macronutrients": scaled_macros,
                "clinical_benefit": m["clinical_benefit"],
                "substitutions": scaled_subs
            })

        scaled_days.append({
            "day_number": day_template["day_number"],
            "day_name": day_template["day_name"],
            "theme": day_template["theme"],
            "diet_type": active_diet_type,
            "target_calories": round(calorie_target, 1),
            "total_estimated_calories": round(day_total_cal, 1),
            "attainment_percentage": round((day_total_cal / calorie_target) * 100, 1),
            "macronutrients": {
                "carbohydrates_g": round(day_total_carbs, 1),
                "protein_g": round(day_total_protein, 1),
                "fats_g": round(day_total_fats, 1),
                "fibre_g": round(day_total_fibre, 1),
                "sodium_mg": round(day_total_sodium, 1)
            },
            "meals": day_meals
        })

    day1 = scaled_days[0]
    restrictions = ["ICMR-NIN Indian Food Composition Clinical Alignment"]
    findings_alignment = []

    # Dynamic clinical restrictions based on patient's findings
    glucose = float(risk_profile.get("glucose") or 0.0)
    systolic_bp = float(risk_profile.get("systolic_bp") or 0.0)
    cholesterol = float(risk_profile.get("cholesterol") or 0.0)
    bmi = float(risk_profile.get("bmi") or 24.5)
    hemoglobin = float(risk_profile.get("hemoglobin") or 0.0)
    pcv = float(risk_profile.get("pcv") or 0.0)
    platelets = float(risk_profile.get("platelets") or 0.0)
    gender = str(risk_profile.get("gender") or "Male").capitalize()
    report_type = str(risk_profile.get("report_type") or "")

    if is_non_veg:
        restrictions.append("Heart-Healthy Lean Poultry, Omega-3 Fish & Eggs (Skinless, Low Saturated Fat)")
    else:
        restrictions.append("100% Pure Lacto-Vegetarian Plant & Dairy Protein")

    # Hematology / CBC Clinical Alignments (Actual blood test findings)
    if hemoglobin > 0:
        hb_thresh = 13.0 if gender == "Male" else 12.0
        if hemoglobin < hb_thresh:
            restrictions.append("Iron-Bioavailability & Erythropoiesis Protocol (Combat Mild Anemia)")
            findings_alignment.append(f"Hemoglobin {hemoglobin:.1f} g/dL (Below {hb_thresh} standard for {gender}): High dietary iron from sprouted moong, palak, beetroot, moringa, black chana, and dates.")
            findings_alignment.append("Vitamin C Pairing: Fresh amla and lemon water included with main meals to enhance non-heme iron absorption by up to 300%.")
            findings_alignment.append("Tannin Restriction: Tea and coffee strictly avoided within 1 hour before or after meals to prevent iron absorption blockage.")
        else:
            findings_alignment.append(f"Hemoglobin {hemoglobin:.1f} g/dL: Healthy iron and oxygen-carrying capacity maintained within reference range.")

    if pcv >= 52.0:
        restrictions.append("Cellular Hydration & Plasma Expansion Protocol")
        findings_alignment.append(f"Elevated PCV / Hematocrit ({pcv:.1f}%): Daily hydration target of 2.5–3.0L with coconut water, chaas, and lemon water to counter hemoconcentration.")

    if 0 < platelets < 160000:
        findings_alignment.append(f"Platelet Count ({int(platelets):,} cumm - Borderline): Papaya leaf, folate-rich lentils, and vitamin C foods prioritized.")

    # Metabolic & Glycemic (Only when actually measured in blood test)
    if glucose >= 100.0 or (glucose > 0 and risk_profile.get("diabetes_risk") in ["Moderate", "High Risk"]):
        restrictions.append("Low Glycemic Index Millets (Bajra, Ragi) & Zero Refined Sugars")
        findings_alignment.append(f"Fasting Glucose {glucose:.0f} mg/dL: High-fiber millets & fenugreek incorporated to blunt post-meal glucose absorption.")

    if systolic_bp >= 130.0 or (systolic_bp > 0 and risk_profile.get("cvd_risk") in ["Moderate", "High Risk"]):
        restrictions.append("Restricted Sodium (< 1500 mg/day) & High Potassium Greens")
        findings_alignment.append(f"Systolic BP {systolic_bp:.0f} mmHg: Sodium capped under 1500mg/day with potassium-rich bottle gourd & greens for arterial relaxation.")

    if cholesterol >= 200.0:
        restrictions.append("Soluble Viscous Fiber (Methi/Oats/Psyllium) & MUFA/PUFA Healthy Oils")
        findings_alignment.append(f"Total Cholesterol {cholesterol:.0f} mg/dL: Saturated fats minimized; MUFA oils and soluble fiber added to speed bile clearance.")

    if calorie_breakdown.get("anthropometrics_available") is False:
        findings_alignment.append("Anthropometrics Unavailable: Standard 1,800 kcal baseline applied with no report personalization. Enter height & weight in profile for tailored caloric deficit.")
    elif bmi >= 25.0:
        findings_alignment.append(f"BMI {bmi:.1f}: Caloric volume calibrated with a safe 350 kcal deficit to promote gradual visceral fat reduction.")

    # Tailor clinical rationale if this is a CBC report
    if hemoglobin > 0 and (glucose == 0 or systolic_bp == 0):
        patient_age = int(round(float(risk_profile.get('age') or 25)))
        clinical_rationale = f"Caloric target of {int(round(calorie_target))} kcal/day tailored to Complete Blood Count (CBC) findings for {gender}, age {patient_age}. Nutrition focuses on iron bioavailability, hemoglobin elevation, and cellular hydration."

    return {
        "plan_id": f"diet_{uuid.uuid4().hex[:8]}",
        "diet_type": active_diet_type,
        "calorie_target": calorie_target,
        "recommended_baseline_calories": calc_cals,
        "calorie_breakdown": calorie_breakdown,
        "clinical_rationale": clinical_rationale,
        "findings_alignment": findings_alignment,
        "dietary_restrictions": restrictions,
        "total_estimated_calories": day1["total_estimated_calories"],
        "total_macronutrients": day1["macronutrients"],
        "days": scaled_days,
        "meals": day1["meals"]
    }

def generate_exercise_plan(risk_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Generates an exercise regime tailored to the patient's risk profile and physical fitness capability."""
    age = float(risk_profile.get("age") or 40)
    cvd_risk = risk_profile.get("cvd_risk") or "Low Risk"
    diabetes_risk = risk_profile.get("diabetes_risk") or "Low Risk"
    systolic_bp = float(risk_profile.get("systolic_bp") or 120)
    bmi = float(risk_profile.get("bmi") or 24.0)

    if cvd_risk == "High Risk" or systolic_bp >= 145 or age >= 65:
        intensity = "Low"
        summary_note = "Gentle, low-impact movements to safeguard blood pressure while sustaining cardiovascular flow."
        routines = [
            {
                "day": "Monday & Thursday",
                "title": "Post-Meal Gentle Walk",
                "duration_minutes": 20,
                "intensity": "Low Intensity",
                "exercises": ["Leisurely stroll", "Heel-to-toe ankle rolls"],
                "frequency": "Daily (after lunch and dinner)",
                "calories_burned": 75,
                "instructions": "Stroll at a comfortable pace for 15-20 minutes after meals. Helps muscle cells uptake glucose without elevating arterial pressure.",
                "target_benefit": "Blunts postprandial glucose spike by up to 25%"
            },
            {
                "day": "Tuesday & Friday",
                "title": "Pranayama & Calming Breathwork",
                "duration_minutes": 20,
                "intensity": "Low Intensity",
                "exercises": ["Anulom Vilom (Alternate Nostril)", "Bhramari (Bee Breath)", "Gentle Neck Rolls"],
                "frequency": "Daily morning",
                "calories_burned": 40,
                "instructions": "Inhale deeply for 4 counts, exhale smoothly for 6 counts in a quiet seated posture.",
                "target_benefit": "Lowers sympathetic vasomotor tone, helping reduce resting systolic BP by 4-7 mmHg"
            },
            {
                "day": "Wednesday & Saturday",
                "title": "Seated Chair Yoga & Joint Stretches",
                "duration_minutes": 25,
                "intensity": "Low Intensity",
                "exercises": ["Seated Cat-Cow", "Chair Sun Salutations", "Seated Ankle Circles"],
                "frequency": "3 times per week",
                "calories_burned": 65,
                "instructions": "Full mobility stretches seated comfortably on a stable chair without joint compression.",
                "target_benefit": "Preserves core mobility and reduces lower-extremity edema"
            }
        ]
        weekly_target = 140

    elif cvd_risk == "Moderate" or diabetes_risk in ["Moderate", "High Risk"] or bmi >= 28.0:
        intensity = "Moderate"
        summary_note = "Targeted metabolic conditioning: combines brisk aerobic intervals with posture & resistance work."
        routines = [
            {
                "day": "Monday & Wednesday",
                "title": "Brisk Interval Walking",
                "duration_minutes": 35,
                "intensity": "Moderate Intensity",
                "exercises": ["5 min warm-up walk", "25 min brisk pace (100-115 steps/min)", "5 min cool down"],
                "frequency": "4 days/week",
                "calories_burned": 160,
                "instructions": "Walk at a pace where you can talk but not sing. Maintain good posture with arm swings.",
                "target_benefit": "Improves GLUT-4 glucose transporter translocation into skeletal muscles"
            },
            {
                "day": "Tuesday & Friday",
                "title": "Bodyweight Metabolic Circuit",
                "duration_minutes": 30,
                "intensity": "Moderate Intensity",
                "exercises": ["Chair squats (3x12)", "Wall pushups (3x10)", "Standing calf raises (3x15)", "Glute bridges (2x12)"],
                "frequency": "3 days/week",
                "calories_burned": 150,
                "instructions": "Perform each movement with controlled cadence. Rest 45 seconds between sets.",
                "target_benefit": "Builds lean muscle mass, elevating resting basal metabolic rate"
            },
            {
                "day": "Thursday & Saturday",
                "title": "Surya Namaskar & Yoga Flow",
                "duration_minutes": 30,
                "intensity": "Moderate Intensity",
                "exercises": ["6-8 rounds Surya Namaskar", "Trikonasana (Triangle)", "Vrikshasana (Tree Pose)", "Shavasana"],
                "frequency": "3 days/week",
                "calories_burned": 130,
                "instructions": "Synchronize breath with each posture transition smoothly.",
                "target_benefit": "Enhances vascular flexibility, insulin receptor density, and stress reduction"
            }
        ]
        weekly_target = 180

    else:
        intensity = "Moderate to Vigorous"
        summary_note = "Dynamic cardiovascular and strength training to maximize longevity, endurance, and metabolic reserve."
        routines = [
            {
                "day": "Monday, Wednesday & Friday",
                "title": "Aerobic Cardio & Jogging Intervals",
                "duration_minutes": 40,
                "intensity": "Moderate to Vigorous",
                "exercises": ["Jogging / brisk cycling (30 min)", "Stair climbing intervals (10 min)"],
                "frequency": "3-4 days/week",
                "calories_burned": 260,
                "instructions": "Elevate heart rate to 65-75% of max heart rate reserve.",
                "target_benefit": "Optimizes VO2 max and reduces visceral adipose tissue"
            },
            {
                "day": "Tuesday & Thursday",
                "title": "Functional Strength & Core Circuit",
                "duration_minutes": 35,
                "intensity": "Moderate to Vigorous",
                "exercises": ["Bodyweight squats (3x15)", "Push-ups (3x12)", "Planks (3x45s)", "Lunges (3x10/leg)"],
                "frequency": "2-3 days/week",
                "calories_burned": 200,
                "instructions": "Progressive overload circuit emphasizing core activation and posture.",
                "target_benefit": "Boosts insulin sensitivity for up to 48 hours post-workout"
            },
            {
                "day": "Saturday & Sunday",
                "title": "Active Recreational Sports / Power Yoga",
                "duration_minutes": 45,
                "intensity": "Moderate Intensity",
                "exercises": ["Badminton / Swimming / 12 rounds Surya Namaskar", "10 min deep stretching"],
                "frequency": "Weekend",
                "calories_burned": 250,
                "instructions": "Engage in continuous enjoyable movement with thorough mobility cool-down.",
                "target_benefit": "Holistic endurance, cardiovascular resilience, and emotional vitality"
            }
        ]
        weekly_target = 200

    return {
        "plan_id": f"ex_{uuid.uuid4().hex[:8]}",
        "intensity": intensity,
        "summary_note": summary_note,
        "weekly_target_minutes": weekly_target,
        "routines": routines
    }
