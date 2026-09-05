const db = require('../db');

async function getProfile(req, res) {
  try {
    const result = await db.query('SELECT * FROM health_profiles WHERE user_id = $1', [req.user.user_id]);
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Health profile not found' });
    }
    const profile = result.rows[0];

    // Get user details (name, email)
    const userRes = await db.query('SELECT name, email, role FROM users WHERE user_id = $1', [req.user.user_id]);
    const userData = userRes.rows[0] || req.user || {};

    const lifestyle = typeof profile.lifestyle_factors === 'string' ? JSON.parse(profile.lifestyle_factors || '{}') : (profile.lifestyle_factors || {});

    return res.json({
      ...profile,
      diet_preference: lifestyle.diet_preference || 'veg',
      lifestyle_factors: lifestyle,
      name: userData.name || req.user.name || '',
      email: userData.email || req.user.email || ''
    });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to retrieve profile', details: err.message });
  }
}

async function updateProfile(req, res) {
  try {
    const { name, age, gender, height, weight, lifestyle_factors, vitals, diet_preference } = req.body;
    
    // Update user name if supplied
    if (name) {
      await db.query('UPDATE users SET name = $1 WHERE user_id = $2', [name, req.user.user_id]);
      req.user.name = name;
    }

    // Merge diet_preference into lifestyle_factors
    let mergedLifestyle = typeof lifestyle_factors === 'object' ? { ...lifestyle_factors } : {};
    if (typeof lifestyle_factors === 'string') {
      try { mergedLifestyle = JSON.parse(lifestyle_factors || '{}'); } catch {}
    }
    if (diet_preference) {
      mergedLifestyle.diet_preference = diet_preference;
    }

    // Calculate BMI if height and weight provided
    let calculatedBmi = null;
    if (height && weight && Number(height) > 0) {
      const h_m = Number(height) / 100;
      calculatedBmi = Number((Number(weight) / (h_m * h_m)).toFixed(2));
    }

    const updatedResult = await db.query(
      `UPDATE health_profiles 
       SET age = $1, gender = $2, height = $3, weight = $4, bmi = $5, lifestyle_factors = $6, vitals = $7, updated_at = $8
       WHERE user_id = $9 RETURNING *`,
      [
        age,
        gender,
        height,
        weight,
        calculatedBmi,
        JSON.stringify(mergedLifestyle),
        typeof vitals === 'object' ? JSON.stringify(vitals) : vitals,
        new Date().toISOString(),
        req.user.user_id
      ]
    );

    const updatedProfile = updatedResult.rows[0] || {};

    return res.json({
      message: 'Health profile updated successfully',
      profile: {
        ...updatedProfile,
        diet_preference: mergedLifestyle.diet_preference || 'veg',
        lifestyle_factors: mergedLifestyle,
        name: name || req.user.name || '',
        email: req.user.email || ''
      }
    });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to update profile', details: err.message });
  }
}

module.exports = {
  getProfile,
  updateProfile
};
