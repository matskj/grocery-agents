@echo off
setlocal

python tools\run_training_run.py %*
exit /b %errorlevel%

