const express = require('express');
const router = express.Router();
const multer = require('multer');
const reportController = require('../controllers/reportController');
const { requireAuth } = require('../middleware/auth');

const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 } // 25 MB max limit
});

const handleUpload = [
  requireAuth,
  (req, res, next) => {
    upload.single('report')(req, res, (err) => {
      if (err) {
        console.error('Multer upload error:', err.message);
        return res.status(400).json({ error: 'File upload error: ' + err.message });
      }
      next();
    });
  },
  reportController.uploadReport
];

const handleMetabolicUpload = [
  requireAuth,
  (req, res, next) => {
    upload.single('report')(req, res, (err) => {
      if (err) {
        console.error('Multer upload error:', err.message);
        return res.status(400).json({ error: 'File upload error: ' + err.message });
      }
      next();
    });
  },
  reportController.uploadMetabolicReport
];

router.post('/upload', ...handleUpload);
router.post('/metabolic-upload', ...handleMetabolicUpload);
router.post('/', ...handleUpload);
router.get('/', requireAuth, reportController.getReports);
router.delete('/all', requireAuth, reportController.deleteAllReports);
router.delete('/:reportId', requireAuth, reportController.deleteReport);

module.exports = router;
