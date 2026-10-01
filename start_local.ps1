Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting LifeMap AI Full Stack Locally" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$root = $PSScriptRoot

Write-Host "`n[1/3] Starting Python FastAPI ML Service (Port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\ml_service'; python -m uvicorn app:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 2

Write-Host "[2/3] Starting Node.js Express Backend (Port 5000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; npm run dev"

Start-Sleep -Seconds 2

Write-Host "[3/3] Starting Next.js Frontend (Port 3000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "LifeMap AI services launched!" -ForegroundColor Green
Write-Host "  Frontend:   http://localhost:3000" -ForegroundColor White
Write-Host "  Backend:    http://localhost:5000" -ForegroundColor White
Write-Host "  ML Service: http://localhost:8000" -ForegroundColor White
Write-Host "========================================================`n" -ForegroundColor Green
