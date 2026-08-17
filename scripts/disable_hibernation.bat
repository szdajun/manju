@echo off
REM disable_hibernation.bat
REM Turn off hibernation to free ~13GB on C: (removes C:\hiberfil.sys).
REM Also disables Fast Startup (it depends on hibernation).
REM Self-elevates: just run it, click YES on the UAC prompt. 2026-08-04.

REM --- self-elevation check ---
net session >nul 2>&1
if errorlevel 1 (
    echo Requesting administrator privileges (click YES on the UAC prompt)...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

REM --- running elevated below ---
echo Disabling hibernation (powercfg /h off)...
powercfg /h off
if errorlevel 1 (
    echo [FAILED] powercfg returned an error. Run this script as administrator.
) else (
    echo [OK] Hibernation disabled. C:\hiberfil.sys removed (~13GB freed).
)
echo.
echo Remaining hibernation settings:
powercfg /a
echo.
echo C: free space now:
fsutil volume diskfree C:
echo.
echo You can close this window.
pause
