@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Building Cython extension...
D:\anaconda3_202111\envs\py385\python.exe setup_cy.py build_ext --inplace
if %ERRORLEVEL% equ 0 (
    echo Build succeeded.
) else (
    echo Build failed. Exit code: %ERRORLEVEL%
)
pause
