const express = require('express');
const router = express.Router();
const recommendationController = require('../controllers/recommendationController');
const { requireAuth } = require('../middleware/auth');

router.post('/generate', requireAuth, recommendationController.generateRecommendations);
router.get('/current', requireAuth, recommendationController.getCurrentPlan);

module.exports = router;
