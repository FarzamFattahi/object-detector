@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Follow the installation steps in README.md.
  pause
  exit /b 1
)
set "YOLO_CONFIG_DIR=%CD%"
".venv\Scripts\python.exe" scripts\download_ppe_model.py
if errorlevel 1 (
  echo Model verification failed. Check the connection and try again.
  pause
  exit /b 1
)
echo Open http://localhost:8501 in your browser.
echo Keep this window open. Press Ctrl+C to stop the demo.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
