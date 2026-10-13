const express = require('express');
const router = express.Router();
const multer = require('multer');
const reportController = require('../controllers/reportController');
const { requireAuth } = require('../middleware/auth');
const { uploadRateLimiter } = require('../middleware/rateLimiter');

const storage = multer.memoryStorage();
const fileFilter = (req, file, cb) => {
  const allowedMimeTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
  const ext = (file.originalname || '').toLowerCase();
  const allowedExtensions = ['.pdf', '.png', '.jpg', '.jpeg'];
  const isValidMime = allowedMimeTypes.includes(file.mimetype);
  const isValidExt = allowedExtensions.some(e => ext.endsWith(e));

  if (isValidMime || isValidExt) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only PDF, PNG, and JPEG documents are permitted.'), false);
  }
};

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB max limit
  fileFilter
});

const handleUpload = [
  requireAuth,
  uploadRateLimiter,
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
  uploadRateLimiter,
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
