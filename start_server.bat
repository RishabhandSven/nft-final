@echo off
REM Start the Wash Trading API server
cd /d D:\nft2\NFT
echo [+] Starting Wash Trading Detection API on port 8001...
.venv\Scripts\python.exe -c "from ai_engine.api import app; from uvicorn import run; run(app, host='127.0.0.1', port=8001, log_level='info')"
if errorlevel 1 (
    echo [!] Server failed to start
    pause
)
