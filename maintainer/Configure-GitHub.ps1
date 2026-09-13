[CmdletBinding()]
param(
    [string]$Owner,
    [string]$Repository = 'SubHooper'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$configPath = Join-Path $root 'app\gui\src-tauri\tauri.conf.json'
$publicKeyPath = Join-Path $root 'private\SubHooper.key.pub'

if ([string]::IsNullOrWhiteSpace($Owner)) { $Owner = Read-Host 'GitHub account name' }
if ($Owner -notmatch '^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$') {
    throw 'The GitHub account name is invalid.'
}
if ($Repository -notmatch '^[A-Za-z0-9._-]+$') { throw 'The GitHub repository name is invalid.' }
if (-not (Test-Path -LiteralPath $publicKeyPath -PathType Leaf)) {
    throw 'Run Generate-Updater-Key.cmd first.'
}

$publicKey = ([System.IO.File]::ReadAllText($publicKeyPath)).Trim()
if ([string]::IsNullOrWhiteSpace($publicKey)) { throw 'The updater public key is empty.' }
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$config.plugins.updater.pubkey = $publicKey
$config.plugins.updater.endpoints = @("https://github.com/$Owner/$Repository/releases/latest/download/latest.json")
$json = $config | ConvertTo-Json -Depth 30
[System.IO.File]::WriteAllText($configPath, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))

Write-Host "Configured updater endpoint for $Owner/$Repository." -ForegroundColor Green

