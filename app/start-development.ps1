[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$readyMarker = Join-Path $PSScriptRoot '.gui-ready-v0.3.6'
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
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    Push-Location (Join-Path $PSScriptRoot 'gui')
    try {
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $npm run tauri dev
            $guiCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousPreference
        }
    } finally {
        Pop-Location
    }
    exit $guiCode
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
