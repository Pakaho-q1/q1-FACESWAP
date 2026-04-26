@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "ROOT=%~dp0"
set "GUI=%ROOT%gui"
call env\Scripts\activate

cd %GUI%
python main.py
pause
