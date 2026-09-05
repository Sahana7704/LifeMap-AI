const express = require('express');
const router = express.Router();
const wellnessController = require('../controllers/wellnessController');
const { requireAuth } = require('../middleware/auth');

router.post('/log', requireAuth, wellnessController.addLog);
router.get('/logs', requireAuth, wellnessController.getLogs);
router.get('/trends', requireAuth, wellnessController.getTrends);

module.exports = router;
