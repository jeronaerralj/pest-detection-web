@echo off
:: This automatically navigates to the folder exactly where this .bat file is located
cd /d "%~dp0"

:: Activate the Python virtual environment
call .venv\Scripts\activate 

:: The /affinity 8 command forces Windows to ONLY use CPU Core #3
start /wait /affinity 8 python train_update.py