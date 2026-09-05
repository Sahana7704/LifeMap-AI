-- LifeMap AI - Database Schema matching Review-1 UML Class Diagram

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'patient',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS health_profiles (
    profile_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    age INTEGER DEFAULT 35,
    gender VARCHAR(20) DEFAULT 'Male',
    height NUMERIC(6, 2) DEFAULT 172.0,
    weight NUMERIC(6, 2) DEFAULT 70.0,
    bmi NUMERIC(5, 2) DEFAULT 23.66,
    lifestyle_factors JSONB DEFAULT '{}',
    vitals JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS medical_reports (
    report_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_image_url TEXT,
    ocr_text TEXT,
    extracted_metrics JSONB DEFAULT '{}',
    extraction_confidence NUMERIC(4, 2) DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    prediction_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    disease_type VARCHAR(100) NOT NULL,
    risk_score NUMERIC(5, 2) NOT NULL,
    risk_category VARCHAR(50) NOT NULL,
    confidence NUMERIC(4, 2) NOT NULL,
    model_version VARCHAR(100),
    vitality_score INTEGER DEFAULT 75,
    input_vitals JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shap_explanations (
    explanation_id VARCHAR(64) PRIMARY KEY,
    prediction_id VARCHAR(64) REFERENCES risk_predictions(prediction_id) ON DELETE CASCADE,
    top_features JSONB DEFAULT '[]',
    risk_increasing_factors JSONB DEFAULT '[]',
    protective_factors JSONB DEFAULT '[]',
    all_contributions JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS diet_plans (
    plan_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    calorie_target NUMERIC(7, 2) NOT NULL,
    total_estimated_calories NUMERIC(7, 2),
    total_macronutrients JSONB DEFAULT '{}',
    dietary_restrictions JSONB DEFAULT '[]',
    meals JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exercise_plans (
    plan_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    intensity VARCHAR(50) NOT NULL,
    summary_note TEXT,
    weekly_target_minutes INTEGER DEFAULT 150,
    routines JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wellness_logs (
    log_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    metrics JSONB NOT NULL DEFAULT '{}',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON risk_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_wellness_user ON wellness_logs(user_id, date);
