@echo off
rem Double-click to open the tracker menu.
cd /d "%~dp0"
set "UV=uv"
where uv >nul 2>nul || set "UV=%USERPROFILE%\.local\bin\uv.exe"
"%UV%" run main.py
if errorlevel 1 pause
