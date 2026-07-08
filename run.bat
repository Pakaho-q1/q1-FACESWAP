@echo off
setlocal EnableExtensions
chcp 65001 >nul
call conda activate faceswap

python faceswap.py gui
pause
