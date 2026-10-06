[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$env:VSLANG = '1033'
$readyMarker = Join-Path $PSScriptRoot '.gui-ready-v0.4.3'
$homeRoot = Split-Path $PSScriptRoot -Parent
$reportsRoot = if ($env:SUBTITLE_REPORTS_ROOT) {
    [System.IO.Path]::GetFullPath($env:SUBTITLE_REPORTS_ROOT)
} elseif ($env:SUBTITLE_HOME) {
    [System.IO.Path]::Combine($env:SUBTITLE_HOME, 'reports')
} else {
    [System.IO.Path]::Combine($PSScriptRoot, 'reports')
}
[System.IO.Directory]::CreateDirectory($reportsRoot) | Out-Null
$env:SUBTITLE_REPORTS_ROOT = $reportsRoot
$env:SUBHOOPER_STARTUP_TRACE = Join-Path $reportsRoot 'startup-0.4.3.log'
Write-Host "Startup trace: $env:SUBHOOPER_STARTUP_TRACE"

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = @(
        (Join-Path $env:USERPROFILE '.cargo\bin'),
        (Join-Path $env:ProgramFiles 'nodejs'),
        $machinePath,
        $userPath
    ) -join ';'
}

function Set-CargoTargetDirectory {
    $env:CARGO_TARGET_DIR = Join-Path $env:LOCALAPPDATA 'SubHooper\build-cache\tauri-v2'
    New-Item -ItemType Directory -Path $env:CARGO_TARGET_DIR -Force | Out-Null
}

function Test-CompatibleNode {
    $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
    if (-not $nodeCommand) { return $false }
    try {
        $version = [version]((& $nodeCommand.Source --version).Trim().TrimStart('v'))
        return (($version.Major -eq 20 -and $version -ge [version]'20.19.0') -or
                ($version -ge [version]'22.12.0'))
    } catch {
        return $false
    }
}

function Get-DevelopmentPort {
    param([Parameter(Mandatory=$true)][int]$PreferredPort)

    $loopback = [System.Net.IPAddress]::Loopback
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new($loopback, $PreferredPort)
        $listener.Start()
        return $PreferredPort
    } catch {
        $preferredError = $_.Exception.Message
        Write-Host "Cannot bind 127.0.0.1:$PreferredPort ($preferredError); selecting an OS-assigned loopback port." -ForegroundColor Yellow
    } finally {
        if ($listener) { $listener.Stop() }
    }

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new($loopback, 0)
        $listener.Start()
        return [int]$listener.LocalEndpoint.Port
    } catch {
        throw "Could not bind the preferred development port $PreferredPort or request an OS-assigned loopback port: $($_.Exception.Message)"
    } finally {
        if ($listener) { $listener.Stop() }
    }
}

try {
    $releaseExe = Join-Path $homeRoot 'SubHooper.exe'
    if (Test-Path -LiteralPath $releaseExe -PathType Leaf) {
        $env:SUBHOOPER_HOME = $homeRoot
        $env:SUBTITLE_HOME = $homeRoot
        $env:SUBTITLE_PROJECT_ROOT = $PSScriptRoot
        $env:SUBTITLE_RESULTS_ROOT = Join-Path $homeRoot 'results'
        $env:SUBTITLE_REPORTS_ROOT = Join-Path $homeRoot 'reports'
        & $releaseExe
        exit $LASTEXITCODE
    }

    Refresh-ProcessPath
    Set-CargoTargetDirectory
    $needsSetup = -not (Test-Path -LiteralPath $readyMarker -PathType Leaf)
    $needsSetup = $needsSetup -or -not (Test-CompatibleNode)
    $needsSetup = $needsSetup -or -not (Get-Command npm.cmd -ErrorAction SilentlyContinue)
    $needsSetup = $needsSetup -or -not (Get-Command cargo.exe -ErrorAction SilentlyContinue)
    if ($needsSetup) {
        & (Join-Path $PSScriptRoot 'setup-development.ps1')
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Refresh-ProcessPath
    }

    $env:SUBTITLE_PROJECT_ROOT = $PSScriptRoot
    $previousDevPort = $env:SUBHOOPER_DEV_PORT
    $configPath = $null
    $locationPushed = $false
    try {
        $devPort = Get-DevelopmentPort -PreferredPort 1420
        $devUrl = "http://127.0.0.1:$devPort"
        $env:SUBHOOPER_DEV_PORT = [string]$devPort
        Write-Host "Development server: $devUrl (loopback only)"
        $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
        $configPath = Join-Path ([System.IO.Path]::GetTempPath()) ("subhooper-tauri-dev-{0}.json" -f [guid]::NewGuid().ToString('N'))
        $tauriConfigPath = Join-Path $PSScriptRoot 'gui\src-tauri\tauri.conf.json'
        $baseCsp = (Get-Content -LiteralPath $tauriConfigPath -Raw | ConvertFrom-Json).app.security.csp
        $devCsp = [System.Text.RegularExpressions.Regex]::new('(connect-src\s+[^;]+)').Replace(
            $baseCsp,
            ('$1 ws://127.0.0.1:' + $devPort),
            1
        )
        if ($devCsp -eq $baseCsp) { throw 'Could not add the selected loopback port to the development CSP.' }
        $devConfig = [ordered]@{
            build = @{ devUrl = $devUrl }
            app = @{ security = @{ devCsp = $devCsp } }
        } | ConvertTo-Json -Depth 8 -Compress
        [System.IO.File]::WriteAllText($configPath, $devConfig, [System.Text.UTF8Encoding]::new($false))
        Push-Location (Join-Path $PSScriptRoot 'gui')
        $locationPushed = $true
        try {
            $previousPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                & $npm run tauri dev -- --config $configPath
                $guiCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousPreference
            }
        } finally {
            if ($locationPushed) { Pop-Location }
        }
    } finally {
        if ($configPath) { Remove-Item -LiteralPath $configPath -Force -ErrorAction SilentlyContinue }
        if ($null -eq $previousDevPort) {
            Remove-Item Env:SUBHOOPER_DEV_PORT -ErrorAction SilentlyContinue
        } else {
            $env:SUBHOOPER_DEV_PORT = $previousDevPort
        }
    }
    if ($guiCode -ne 0) {
        Write-Host "Development startup exited with code $guiCode. Vite and Tauri were configured for $devUrl; see the startup trace and GUI output for the bind failure details." -ForegroundColor Red
    }
    exit $guiCode
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
