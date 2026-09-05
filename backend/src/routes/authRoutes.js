const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const { requireAuth } = require('../middleware/auth');

router.post('/register', authController.register);
router.post('/login', authController.login);
router.get('/accounts', authController.getRecentAccounts);
router.get('/me', requireAuth, authController.getMe);

module.exports = router;
