@echo off
REM compact_wsl_vhdx.bat — shrink F:\WSL\Ubuntu\ext4.vhdx (reclaim unused interior space).
REM Needs Administrator (diskpart). Self-elevates: just double-click, click YES on UAC. 2026-08-04.
REM Safe: compact only reclaims already-freed blocks inside the vhdx; no data loss.
REM Side effect: stops Docker Desktop + WSL briefly; restart Docker after if you need it.

REM --- self-elevation check ---
net session >nul 2>&1
if errorlevel 1 (
    echo Requesting administrator privileges (click YES on the UAC prompt)...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

REM --- running elevated below ---
echo === [1/4] Stopping Docker Desktop (so it releases the vhdx) ===
taskkill /f /im "Docker Desktop.exe" 2>nul
taskkill /f /im "com.docker.backend.exe" 2>nul
taskkill /f /im "com.docker.service" 2>nul

echo === [2/4] Shutting down WSL ===
wsl --shutdown
echo Waiting 8s for the vhdx file handle to release...
timeout /t 8 /nobreak >nul

echo === [3/4] Current vhdx size ===
dir "F:\WSL\Ubuntu\ext4.vhdx"

echo === [4/4] Compacting vhdx via diskpart (takes 1-5 min) ===
set DPS=%TEMP%\dp_compact.txt
> "%DPS%" (
  echo select vdisk file="F:\WSL\Ubuntu\ext4.vhdx"
  echo attach vdisk readonly
  echo compact vdisk
  echo detach vdisk
)
diskpart /s "%DPS%"
del "%DPS%" 2>nul

echo.
echo === DONE. New vhdx size: ===
dir "F:\WSL\Ubuntu\ext4.vhdx"
echo.
echo If attach failed with "access denied", Docker/WSL was still holding the file -
echo close Docker Desktop fully from the tray, wait 10s, and re-run this script.
echo.
echo You can close this window. Restart Docker Desktop if you need it.
pause
