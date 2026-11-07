# Record the REAL agent running in its own terminal window with ffmpeg (Windows).
#   .\tools\record_agent.ps1 -Mode baseline -WorkDir <folder with the CSVs> -OutDir <where to put the mp4>
#   .\tools\record_agent.ps1 -Mode pack     -WorkDir <folder with the CSVs> -OutDir <where to put the mp4>
# Only that window's visible rectangle is captured (never the whole screen). Do not touch the mouse or
# keyboard while it runs; a terminal window appears on screen and is closed at the end.
param(
    [Parameter(Mandatory = $true)][ValidateSet('baseline', 'pack')][string]$Mode,
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$TimeoutSec = 900
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force $OutDir | Out-Null
$title = "honest-ml-agent-$Mode"
$go = Join-Path $OutDir "go_$Mode.flag"
$done = Join-Path $OutDir "done_$Mode.flag"
$mp4 = Join-Path $OutDir "$Mode.mkv"      # ffmpeg is stopped gracefully, so the file is complete
Remove-Item $go, $done, $mp4 -ErrorAction SilentlyContinue

Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public struct RECT { public int L, T, R, B; }
public class Win {
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int hh, bool r);
  [DllImport("user32.dll")] public static extern int GetSystemMetrics(int i);
  [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h, int attr, out RECT r, int size);
}
'@
[void][Win]::SetProcessDPIAware()

$launcher = Join-Path $OutDir "launch_$Mode.cmd"
@"
@echo off
title $title
mode con: cols=110 lines=34
python "$root\tools\agent_demo.py" --mode $Mode --dir "$WorkDir" --transcript "$OutDir\$Mode.jsonl" --wait-for "$go" --done-file "$done" --hold 5
"@ | Set-Content -Encoding ASCII $launcher

$term = Start-Process cmd -ArgumentList '/c', "`"$launcher`"" -PassThru
$wt = $null
for ($i = 0; $i -lt 50 -and -not $wt; $i++) {
    Start-Sleep -Milliseconds 300
    $wt = Get-Process WindowsTerminal -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*$title*" } | Select-Object -First 1
}
if (-not $wt) { throw "terminal window '$title' did not appear" }
$h = $wt.MainWindowHandle
$sw = [Win]::GetSystemMetrics(0); $sh = [Win]::GetSystemMetrics(1)
$w = [Math]::Min(1280, $sw - 80); $hh = [Math]::Min(760, $sh - 80)
[void][Win]::MoveWindow($h, 40, 20, $w, $hh, $true); [void][Win]::SetForegroundWindow($h)
Start-Sleep -Seconds 2
$r = New-Object RECT
[void][Win]::DwmGetWindowAttribute($h, 9, [ref]$r, 16)       # DWMWA_EXTENDED_FRAME_BOUNDS = exact visible bounds
$cw = $r.R - $r.L; $ch = $r.B - $r.T
if ($cw % 2) { $cw-- }; if ($ch % 2) { $ch-- }
"window: $($r.L),$($r.T) ${cw}x${ch}  screen: ${sw}x${sh}"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = 'ffmpeg'
$psi.Arguments = "-hide_banner -loglevel error -y -f gdigrab -framerate 10 -offset_x $($r.L) -offset_y $($r.T) -video_size ${cw}x${ch} -i desktop -c:v libx264 -preset ultrafast -crf 20 -pix_fmt yuv420p -g 10 `"$mp4`""
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.CreateNoWindow = $true
$ff = [System.Diagnostics.Process]::Start($psi)
Start-Sleep -Seconds 2
Set-Content $go "go"
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while (-not (Test-Path $done) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
Start-Sleep -Seconds 1
$ff.StandardInput.WriteLine('q')                 # graceful stop: ffmpeg finalises the file
if (-not $ff.WaitForExit(20000)) { $ff.Kill() }
Stop-Process -Id $wt.Id -Force -ErrorAction SilentlyContinue
if (Test-Path $done) { "finished: $mp4" } else { "TIMED OUT after $TimeoutSec s: $mp4" }
