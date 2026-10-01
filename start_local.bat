@echo off
echo ========================================================
echo Starting LifeMap AI Full Stack Locally
echo ========================================================
echo.

echo [1/3] Starting Python FastAPI ML Service (Port 8000)...
start "LifeMap AI - ML Service" cmd /k "cd /d %~dp0ml_service && python -m uvicorn app:app --host 127.0.0.1 --port 8000"

timeout /t 2 /nobreak >nul

echo [2/3] Starting Node.js Express Backend (Port 5000)...
start "LifeMap AI - Backend" cmd /k "cd /d %~dp0backend && npm run dev"

timeout /t 2 /nobreak >nul

echo [3/3] Starting Next.js Frontend (Port 3000)...
start "LifeMap AI - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================================
echo LifeMap AI is starting up!
echo   Frontend:   http://localhost:3000
echo   Backend:    http://localhost:5000
echo   ML Service: http://localhost:8000
echo ========================================================
echo.
