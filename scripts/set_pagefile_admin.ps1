#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Fix the LTX-2.3 GGUF pipeline crash (2026-08-04).
  MUST be run as Administrator, THEN REBOOT.

.PROBLEM
  "System managed" pagefile on this machine created NO C:\pagefile.sys
  (Win32_PageFileUsage empty). Commit limit stuck at physical RAM (31.3 GB).
  The LTX GGUF pipeline needs ~58-61 GB commit to dequant the 22B weights
  (comfyui-gguf pin_weight_to_device) -> abort at ~30.7 GB committed.
  Smoke _ltx23_gguf_smoke.py crashed in CLIP load; commit log showed limit
  flat at 31.31 GB for the entire run.

.FIX
  Explicit C:\pagefile.sys, managed initial 16 GB / max 32 GB.
  - 16 GB base at boot GUARANTEES a pagefile exists (the missing ingredient).
  - Grows to 32 GB under GGUF load -> commit limit 31.3+32 = ~63 GB
    (peak ~58 GB aimdo / ~61 GB legacy -> ~2-5 GB headroom, proven path).
  - C: has 39 GB free -> 32 GB max leaves ~7 GB on C.

.ESCALATION
  If the smoke STILL crashes after this, set fixed (initial=max=32768) for a
  full-size pagefile from boot (no growth-timing dependence), accepting C at
  ~7 GB free permanently.

.VERIFY (after reboot, no admin needed):
  powershell -c "Get-CimInstance Win32_PageFileUsage | fl Name,AllocatedBaseSize; \
    [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVirtualMemorySize/1MB,1)"
  Expect AllocatedBaseSize >= 16384 and TotalVirtualMemorySize/1MB >= ~47000.
#>
$ErrorActionPreference = 'Stop'

Write-Host "[1/3] Clearing existing pagefile settings (auto-manage marker)..." -ForegroundColor Cyan
Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-CimInstance $_ }

Write-Host "[2/3] Setting C:\pagefile.sys  initial=16384 MB  max=32768 MB ..." -ForegroundColor Cyan
New-CimInstance Win32_PageFileSetting -Property @{
    Name        = 'C:\pagefile.sys'
    InitialSize = 16384
    MaximumSize = 32768
} | Out-Null

Write-Host "[3/3] Current Win32_PageFileSetting:" -ForegroundColor Cyan
Get-CimInstance Win32_PageFileSetting | Format-List Name, InitialSize, MaximumSize

Write-Host ""
Write-Host ">>> REBOOT NOW to activate the pagefile. <<<" -ForegroundColor Yellow
Write-Host "After reboot, re-run:  ../ComfyUI/venv/Scripts/python.exe  ../manju/scripts/_ltx23_gguf_smoke.py"
