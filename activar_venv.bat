@echo off
cd /d "%~dp0"
call ".\venv\Scripts\activate.bat"
python -c "import sys; print(sys.executable)"
pause

::  .\venv\Scripts\activate