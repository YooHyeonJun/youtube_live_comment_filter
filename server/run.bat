@echo off
setlocal

cd /d %~dp0

REM Default env (can be overridden by caller)
if "%PORT%"=="" set PORT=8000
if "%HF_MODEL_ID%"=="" set HF_MODEL_ID=beomi/KcELECTRA-base-v2022

REM Create/activate venv
if not exist .venv (
	python -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

REM Ensure model dir exists one level up from server/
set MODEL_DIR=%~dp0..\model
if not exist "%MODEL_DIR%" (
	mkdir "%MODEL_DIR%"
)

REM Run app from server directory
python app.py

endlocal


