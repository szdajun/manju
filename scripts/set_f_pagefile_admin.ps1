# ==================================================================
# 在现有 C: 32GB pagefile 之上【叠加】F: 32GB 固定 pagefile
# 顶: 63.3GB -> ~95GB, 让 Q8 冷加载(峰值 64GB)能过 GGUF node:85
#
# 用法: 右键 PowerShell -> "以管理员身份运行" -> 粘贴运行此脚本 -> 重启
# ==================================================================
#Requires -RunAsAdministrator

Write-Host "=== 改前 (期望已有 C:\pagefile.sys 32768) ===" -ForegroundColor Cyan
Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize | Format-Table

# 1. 关掉"自动管理"，让显式设置生效 (C: 已是 fixed, 此项应已 $false, 幂等再设一次)
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile = $false} | Out-Null
    Write-Host "已关闭 AutomaticManagedPagefile" -ForegroundColor Yellow
} else {
    Write-Host "AutomaticManagedPagefile 已是 false (OK)"
}

# 2. 加 F: 32GB 固定 pagefile (不动 C:)
$exists = Get-CimInstance Win32_PageFileSetting | Where-Object { $_.Name -eq 'F:\pagefile.sys' }
if ($null -eq $exists) {
    New-CimInstance Win32_PageFileSetting -Property @{
        Name = 'F:\pagefile.sys'; InitialSize = 32768; MaximumSize = 32768
    } | Out-Null
    Write-Host "已添加 F:\pagefile.sys  32768 / 32768 MB" -ForegroundColor Green
} else {
    Write-Host "F:\pagefile.sys 已存在, 跳过" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 改后 (必须同时有 C: 和 F: 两条, 各 32768) ===" -ForegroundColor Cyan
Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize | Format-Table

Write-Host ""
Write-Host "完成。现在重启:  Restart-Computer" -ForegroundColor Magenta
Write-Host "(重启后 Claude 会验证新顶 >= 95GB, 然后自动跑 Q8 诊断)"
