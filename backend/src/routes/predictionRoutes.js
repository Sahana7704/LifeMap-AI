const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/predictionController');
const { requireAuth } = require('../middleware/auth');

router.post('/run', requireAuth, predictionController.runPrediction);
router.get('/latest', requireAuth, predictionController.getLatestPrediction);
router.get('/history', requireAuth, predictionController.getPredictionHistory);

module.exports = router;
