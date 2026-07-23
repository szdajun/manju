# keep_awake_90min.ps1 - Hold execution state for 90min then release.
# ASCII comments only (PS5.1 GBK parse issue).
# Pattern lifted from manju HANDOFF (kohya training long run safe).
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class Aw {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$hold = [uint32]2147483649  # ES_CONTINUOUS(0x80000000) | ES_SYSTEM_REQUIRED(0x00000001)
$ret  = [Aw]::SetThreadExecutionState($hold)
Write-Host "[keep-awake ON] ret=$ret  ttl=90min  release pid=$$"
Start-Sleep -Seconds 5400
Write-Host "[keep-awake tick 1/2, refreshing]"
$ret  = [Aw]::SetThreadExecutionState($hold)
Start-Sleep -Seconds 5400
Write-Host "[keep-awake OFF at 90min]"
[Aw]::SetThreadExecutionState([uint32]2147483648)  # ES_CONTINUOUS only = release
