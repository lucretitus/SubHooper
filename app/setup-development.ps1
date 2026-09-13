[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$guiRoot = Join-Path $PSScriptRoot 'gui'
$reportsRoot = if ($env:SUBTITLE_REPORTS_ROOT) {
    [System.IO.Path]::GetFullPath($env:SUBTITLE_REPORTS_ROOT)
} else {
    $PSScriptRoot
}
[System.IO.Directory]::CreateDirectory($reportsRoot) | Out-Null
$setupLog = [System.IO.Path]::Combine($reportsRoot, 'gui-setup.log')
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
$env:NO_COLOR = '1'
$env:FORCE_COLOR = '0'

function Invoke-NativeLogged {
    param(
        [Parameter(Mandatory=$true)][string]$Executable,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )
    $previousPreference = $ErrorActionPreference
    $writer = [System.IO.StreamWriter]::new($setupLog, $true, $utf8NoBom)
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1 | ForEach-Object {
            $line = if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.Exception.Message
            } else {
                [string]$_
            }
            Write-Host $line
            $writer.WriteLine($line)
            $writer.Flush()
        }
        $nativeCode = $LASTEXITCODE
    } finally {
        $writer.Dispose()
        $ErrorActionPreference = $previousPreference
    }
    return $nativeCode
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory=$true)][string]$Executable,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )
    $nativeCode = Invoke-NativeLogged $Executable $Arguments
    if ($nativeCode -ne 0) {
        throw "$Executable exited with code $nativeCode"
    }
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $cargoPath = Join-Path $env:USERPROFILE '.cargo\bin'
    $nodePath = Join-Path $env:ProgramFiles 'nodejs'
    $env:Path = @($cargoPath, $nodePath, $machinePath, $userPath) -join ';'
}

function Set-CargoTargetDirectory {
    $env:CARGO_TARGET_DIR = Join-Path $env:LOCALAPPDATA 'SubHooper\build-cache\tauri-v2'
    New-Item -ItemType Directory -Path $env:CARGO_TARGET_DIR -Force | Out-Null
}

function Test-VCTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { return $false }
    $installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    return -not [string]::IsNullOrWhiteSpace(($installation | Out-String))
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

function Test-RustToolchain {
    param([Parameter(Mandatory=$true)][string]$Rustup)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Rustup run stable-msvc rustc -V *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

try {
    [System.IO.File]::WriteAllText(
        $setupLog,
        "Generated=$([DateTimeOffset]::Now.ToString('o'))$([Environment]::NewLine)",
        $utf8NoBom
    )
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw 'Windows Package Manager (winget) was not found. Update Microsoft App Installer.'
    }

    if (-not (Test-CompatibleNode)) {
        Write-Host 'Installing Node.js LTS...'
        Invoke-NativeChecked $winget.Source @(
            'install', '--id', 'OpenJS.NodeJS.LTS', '--exact', '--source', 'winget',
            '--accept-package-agreements', '--accept-source-agreements', '--force'
        )
        Refresh-ProcessPath
        if (-not (Test-CompatibleNode)) {
            throw 'Could not install Node.js 20.19+ or 22.12+.'
        }
    }

    if (-not (Get-Command cargo.exe -ErrorAction SilentlyContinue)) {
        Write-Host 'Installing the Rust MSVC toolchain...'
        Invoke-NativeChecked $winget.Source @(
            'install', '--id', 'Rustlang.Rustup', '--exact', '--source', 'winget',
            '--accept-package-agreements', '--accept-source-agreements'
        )
        Refresh-ProcessPath
    }

    if (-not (Test-VCTools)) {
        Write-Host 'Installing Microsoft C++ Build Tools. Windows may request administrator approval...'
        Invoke-NativeChecked $winget.Source @(
            'install', '--id', 'Microsoft.VisualStudio.2022.BuildTools', '--exact', '--source', 'winget',
            '--accept-package-agreements', '--accept-source-agreements',
            '--override', '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
        )
    }

    Refresh-ProcessPath
    Set-CargoTargetDirectory
    $node = (Get-Command node.exe -ErrorAction Stop).Source
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    $cargo = (Get-Command cargo.exe -ErrorAction Stop).Source
    $rustup = (Get-Command rustup.exe -ErrorAction Stop).Source

    Invoke-NativeChecked $rustup @('default', 'stable-msvc')
    if (-not (Test-RustToolchain $rustup)) {
        Write-Host 'Repairing the incomplete Rust toolchain...'
        $updateCode = Invoke-NativeLogged $rustup @('update', 'stable-msvc', '--no-self-update')
        if ($updateCode -ne 0 -or -not (Test-RustToolchain $rustup)) {
            Write-Host 'Reinstalling the damaged Rust toolchain...'
            $uninstallCode = Invoke-NativeLogged $rustup @(
                'toolchain', 'uninstall', 'stable-x86_64-pc-windows-msvc'
            )
            if ($uninstallCode -ne 0) {
                throw "Could not remove the damaged Rust toolchain: $uninstallCode"
            }
            Invoke-NativeChecked $rustup @(
                'toolchain', 'install', 'stable-msvc', '--profile', 'minimal', '--no-self-update'
            )
            Invoke-NativeChecked $rustup @('default', 'stable-msvc')
        }
        if (-not (Test-RustToolchain $rustup)) {
            throw 'Could not repair the Rust MSVC toolchain.'
        }
    }
    Push-Location $guiRoot
    try {
        Invoke-NativeChecked $npm @('ci', '--no-audit', '--no-fund')
        Invoke-NativeChecked $npm @('test')
        Invoke-NativeChecked $npm @('run', 'build')
        Invoke-NativeChecked $cargo @('check', '--manifest-path', (Join-Path $guiRoot 'src-tauri\Cargo.toml'))
        Invoke-NativeChecked $cargo @('test', '--manifest-path', (Join-Path $guiRoot 'src-tauri\Cargo.toml'))
    } finally {
        Pop-Location
    }

    'READY' | Set-Content -LiteralPath (Join-Path $PSScriptRoot '.gui-ready-v0.3.7') -Encoding ASCII
    Write-Host 'GUI development environment is ready.' -ForegroundColor Green
    Write-Host "Node=$(& $node --version)"
    Write-Host "Cargo=$(& $cargo --version)"
    exit 0
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Setup log: $setupLog" -ForegroundColor Yellow
    exit 1
}
