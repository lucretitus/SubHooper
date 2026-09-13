[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$gui = Join-Path $root 'app\gui'
$configPath = Join-Path $gui 'src-tauri\tauri.conf.json'
$configText = Get-Content -LiteralPath $configPath -Raw

if ($configText.Contains('REPLACE_WITH_')) {
    throw 'Run Generate-Updater-Key.cmd and Configure-GitHub.cmd before building a release.'
}
foreach ($command in @('node.exe', 'npm.cmd', 'cargo.exe', 'rustc.exe')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$command is required for a local maintainer build. GitHub Actions can build without local setup."
    }
}
if (-not $env:TAURI_SIGNING_PRIVATE_KEY -and -not $env:TAURI_SIGNING_PRIVATE_KEY_PATH) {
    $defaultKey = Join-Path $root 'private\SubHooper.key'
    if (-not (Test-Path -LiteralPath $defaultKey -PathType Leaf)) {
        throw 'Set TAURI_SIGNING_PRIVATE_KEY or TAURI_SIGNING_PRIVATE_KEY_PATH.'
    }
    $env:TAURI_SIGNING_PRIVATE_KEY_PATH = $defaultKey
}

Push-Location $gui
try {
    npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
    npm.cmd test
    if ($LASTEXITCODE -ne 0) { throw 'Frontend tests failed.' }
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    cargo.exe test --manifest-path src-tauri\Cargo.toml
    if ($LASTEXITCODE -ne 0) { throw 'Rust tests failed.' }
    npm.cmd run tauri -- build --bundles nsis
    if ($LASTEXITCODE -ne 0) { throw 'Tauri installer build failed.' }
} finally {
    Pop-Location
}

$bundle = Join-Path $gui 'src-tauri\target\release\bundle\nsis'
Write-Host "Installer build complete: $bundle" -ForegroundColor Green

