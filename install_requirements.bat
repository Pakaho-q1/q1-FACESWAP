@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo AI PIPELINE AUTO-SETUP
echo ==========================================

:: 1. เช็กว่ามีโฟลเดอร์ env หรือยัง
if not exist "env" (
    echo [1/3] Virtual environment NOT found.
    echo.
    echo Searching for installed Python versions...
    echo ------------------------------------------
    :: แสดงรายการ Python ในเครื่อง (เวอร์ชันที่มีเครื่องหมาย * คือ default)
    py -0
    echo ------------------------------------------
    
    :: ให้ผู้ใช้เลือกเวอร์ชัน (ใส่ค่าว่างเพื่อใช้ default ของระบบ)
    set /p py_ver="Please enter the version to use (e.g., 3.10) or press ENTER for default: "
    
    echo.
    :: ตรวจสอบว่าผู้ใช้พิมพ์อะไรมาไหม ถ้าว่างเปล่าให้ใช้คำสั่งสร้าง venv แบบปกติ
    if "!py_ver!"=="" (
        echo Creating virtual environment with Default Python...
        py -m venv env
    ) else (
        echo Creating virtual environment with Python !py_ver!...
        py -!py_ver! -m venv env
    )
    
    :: ใช้ !errorlevel! แทน %errorlevel% เพราะอยู่ใต้วงเล็บ
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create venv. Please check if Python version is correct.
        pause
        exit /b
    )
) else (
    echo [1/3] Virtual environment already exists. Skipping creation.
)

:: 2. เปิดใช้งาน Environment
echo.
echo Activating Environment...
call env\Scripts\activate

:: 3. เช็กเวอร์ชัน Python ภายใน env
echo [2/3] Checking Python Version...
python --version

:: 4. อัปเดต pip
echo.
echo [3/3] Updating pip...
python -m pip install --upgrade pip

:: 5. รันสคริปต์ติดตั้งและตรวจสอบ
echo.
echo ==========================================
echo STARTING INSTALLATION AND VERIFICATION
echo ==========================================
python install_and_verify.py

echo.
echo ==========================================
echo SETUP COMPLETE!
echo ==========================================
pause