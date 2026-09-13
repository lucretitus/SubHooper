[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$gui = Join-Path $root 'app\gui'
$private = Join-Path $root 'private'
$key = Join-Path $private 'SubHooper.key'

if (Test-Path -LiteralPath $key -PathType Leaf) {
    throw 'private\SubHooper.key already exists. Existing updater keys must not be replaced.'
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw 'Node.js 22 or newer is required for this one-time maintainer task.'
}

New-Item -ItemType Directory -Path $private -Force | Out-Null
Push-Location $gui
try {
    npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
    npm.cmd run tauri -- signer generate -w $key
    if ($LASTEXITCODE -ne 0) { throw 'Updater key generation failed.' }
} finally {
    Pop-Location
}

Write-Host 'Updater key pair created under private\.' -ForegroundColor Green
Write-Host 'Back up the private key and password. Never commit the private directory.'

