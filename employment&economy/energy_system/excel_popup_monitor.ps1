param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [double]$PollSeconds = 0.5
)

$code = @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class Win32PopupMonitor {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);
}
"@

Add-Type -TypeDefinition $code

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $dir = Split-Path -Parent $LogPath
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    Add-Content -Path $LogPath -Value "[$stamp] $Message"
}

function Get-WindowTextValue {
    param([System.IntPtr]$Hwnd)
    $length = [Win32PopupMonitor]::GetWindowTextLength($Hwnd)
    $builder = New-Object System.Text.StringBuilder ($length + 1)
    [void][Win32PopupMonitor]::GetWindowText($Hwnd, $builder, $builder.Capacity)
    $builder.ToString().Trim()
}

function Get-ClassNameValue {
    param([System.IntPtr]$Hwnd)
    $builder = New-Object System.Text.StringBuilder 256
    [void][Win32PopupMonitor]::GetClassName($Hwnd, $builder, $builder.Capacity)
    $builder.ToString().Trim()
}

function Get-ProcessNameForWindow {
    param([System.IntPtr]$Hwnd)
    $pid = 0
    [void][Win32PopupMonitor]::GetWindowThreadProcessId($Hwnd, [ref]$pid)
    try {
        (Get-Process -Id $pid -ErrorAction Stop).ProcessName
    } catch {
        ""
    }
}

function Close-Popup {
    param([System.IntPtr]$Hwnd)
    [void][Win32PopupMonitor]::PostMessage($Hwnd, 0x0100, [IntPtr]13, [IntPtr]0)
    [void][Win32PopupMonitor]::PostMessage($Hwnd, 0x0101, [IntPtr]13, [IntPtr]0)
    Start-Sleep -Milliseconds 200
    [void][Win32PopupMonitor]::PostMessage($Hwnd, 0x0010, [IntPtr]0, [IntPtr]0)
}

$seen = New-Object System.Collections.Generic.HashSet[string]
Write-Log "excel_popup_monitor.ps1 started"

while ($true) {
    $windows = New-Object System.Collections.Generic.List[object]
    $callback = [Win32PopupMonitor+EnumWindowsProc]{
        param($hWnd, $lParam)
        if (-not [Win32PopupMonitor]::IsWindowVisible($hWnd)) {
            return $true
        }

        $title = Get-WindowTextValue -Hwnd $hWnd
        if ([string]::IsNullOrWhiteSpace($title)) {
            return $true
        }

        $className = Get-ClassNameValue -Hwnd $hWnd
        $processName = Get-ProcessNameForWindow -Hwnd $hWnd
        $isTargetProcess = @("EXCEL", "python", "pythonw") -contains $processName
        $isTargetTitle = @("Error", "Microsoft Excel", "Microsoft Visual Basic", "xlwings") -contains $title
        $isExcelDialog = ($className -eq "#32770" -and $processName -eq "EXCEL")

        if ($isTargetProcess -and ($isTargetTitle -or $isExcelDialog)) {
            $windows.Add([PSCustomObject]@{
                Hwnd = $hWnd
                Title = $title
                ClassName = $className
                ProcessName = $processName
            }) | Out-Null
        }
        return $true
    }

    [void][Win32PopupMonitor]::EnumWindows($callback, [IntPtr]::Zero)

    foreach ($window in $windows) {
        $sig = "$($window.ProcessName)|$($window.ClassName)|$($window.Title)"
        if (-not $seen.Contains($sig)) {
            [void]$seen.Add($sig)
            Write-Log "popup detected: process=$($window.ProcessName) class=$($window.ClassName) title='$($window.Title)'"
        }
        Close-Popup -Hwnd $window.Hwnd
        Write-Log "popup handled: process=$($window.ProcessName) title='$($window.Title)'"
    }

    Start-Sleep -Milliseconds ([int]($PollSeconds * 1000))
}
