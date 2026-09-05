const path = require('path');
const db = require('../backend/src/db');

async function testBackend() {
  console.log('Testing backend db mock updates...');
  
  const testUserId = 'usr_test_' + Date.now();
  
  // 1. Insert a test health profile
  await db.query(
    `INSERT INTO health_profiles 
     (profile_id, user_id, age, gender, height, weight, bmi, lifestyle_factors, vitals, updated_at)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
    [
      'prf_1',
      testUserId,
      40,
      'Female',
      158,
      60,
      24.0,
      JSON.stringify({}),
      JSON.stringify({}),
      new Date().toISOString()
    ]
  );

  // 2. Simulate metabolic report upload updating health profile with Male, 175cm, 84kg
  await db.query(
    `UPDATE health_profiles 
     SET vitals = $1, age = COALESCE($2, age), gender = COALESCE($3, gender), bmi = COALESCE($4, bmi),
         height = COALESCE($5, height), weight = COALESCE($6, weight), updated_at = $7
     WHERE user_id = $8`,
    [
      JSON.stringify({ height_cm: 175, weight_kg: 84, gender: 'Male', age: 45 }),
      45,
      'Male',
      27.4,
      175,
      84,
      new Date().toISOString(),
      testUserId
    ]
  );

  const res = await db.query('SELECT * FROM health_profiles WHERE user_id = $1', [testUserId]);
  const updated = res.rows[0];
  console.log('Updated profile:', {
    age: updated.age,
    gender: updated.gender,
    height: updated.height,
    weight: updated.weight,
    bmi: updated.bmi
  });

  if (updated.height !== 175 || updated.weight !== 84 || updated.gender !== 'Male' || updated.age !== 45) {
    console.error('FAIL: Profile was not updated properly!');
    process.exit(1);
  }

  console.log('SUCCESS: Profile correctly updated with verified report demographics and vitals!');
  process.exit(0);
}

testBackend().catch(err => {
  console.error(err);
  process.exit(1);
});
