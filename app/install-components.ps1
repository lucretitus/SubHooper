[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
if ([string]::IsNullOrWhiteSpace($localAppData)) {
    throw 'Windows LocalAppData could not be resolved.'
}

$componentRoot = [System.IO.Path]::Combine($localAppData, 'SubHooper', 'components')
$downloadRoot = [System.IO.Path]::Combine($localAppData, 'SubHooper', 'downloads')
$runtimeRoot = [System.IO.Path]::Combine($localAppData, 'SubHooper', 'runtime', 'ocr-cpu-py314-auto')
$vsfRoot = [System.IO.Path]::Combine($componentRoot, 'VideoSubFinder-6.10')
$vsfExe = [System.IO.Path]::Combine($vsfRoot, 'Release_x64', 'VideoSubFinderWXW.exe')
$pythonRoot = [System.IO.Path]::Combine($componentRoot, 'Python-3.14.7')
$pythonExe = [System.IO.Path]::Combine($pythonRoot, 'python.exe')
$runtimePython = [System.IO.Path]::Combine($runtimeRoot, 'Scripts', 'python.exe')
$probe = [System.IO.Path]::Combine($PSScriptRoot, 'engine', 'probe.py')

$vsfUrls = @(
    'https://downloads.sourceforge.net/project/videosubfinder/VideoSubFinder_6.10_x64.zip',
    'https://sourceforge.net/projects/videosubfinder/files/VideoSubFinder_6.10_x64.zip/download',
    'https://master.dl.sourceforge.net/project/videosubfinder/VideoSubFinder_6.10_x64.zip?viasf=1'
)
$vsfSha256 = '3c0cc03793ec9753a6a4ee8a91c1d226c20b80aab901718f7c97d4fcb3580c0e'
$pythonUrl = 'https://www.python.org/ftp/python/3.14.7/python-3.14.7-amd64.exe'
$pythonSha256 = '9d9eb2709ef81bf5cd30db3c2096bdbc4ea10087c22e62f27d356b36f6ae9649'
$vcRuntimeUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'

function Invoke-ComponentDownload {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Destination
    )

    $curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
    if ($null -ne $curl) {
        $curlArguments = @(
            '--silent', '--show-error', '--location', '--fail',
            '--retry', '3', '--retry-delay', '2', '--connect-timeout', '30',
            '--output', $Destination, $Url
        )
        & $curl.Source @curlArguments
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe exited with code $LASTEXITCODE."
        }
        return
    }

    Invoke-WebRequest -UseBasicParsing -MaximumRedirection 10 `
        -Headers @{ 'User-Agent' = 'SubHooper/0.3.7' } `
        -Uri $Url -OutFile $Destination
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory=$true)][string[]]$Urls,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][string]$Sha256,
        [Parameter(Mandatory=$true)][string]$Label
    )
    $expected = $Sha256.ToLowerInvariant()
    if ([System.IO.File]::Exists($Destination)) {
        $existing = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existing -eq $expected) { return }
        Remove-Item -LiteralPath $Destination -Force
    }

    $partial = "$Destination.partial"
    $lastFailure = 'No download attempt completed.'
    for ($sourceIndex = 0; $sourceIndex -lt $Urls.Count; $sourceIndex++) {
        $url = $Urls[$sourceIndex]
        foreach ($attempt in 1..3) {
            Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
            Write-Output "Downloading $Label (source $($sourceIndex + 1)/$($Urls.Count), attempt $attempt/3)..."
            try {
                Invoke-ComponentDownload -Url $url -Destination $partial
                if (-not [System.IO.File]::Exists($partial)) {
                    throw 'The downloader did not create a file.'
                }
                $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actual -eq $expected) {
                    Move-Item -LiteralPath $partial -Destination $Destination -Force
                    return
                }
                $length = (Get-Item -LiteralPath $partial).Length
                $lastFailure = "Source returned an unverified file ($length bytes, SHA-256 $actual)."
            } catch {
                $lastFailure = $_.Exception.Message
            } finally {
                Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
            }
            Start-Sleep -Seconds 2
        }
    }

    throw "$Label could not be downloaded and verified after trying all official sources. Expected SHA-256 $expected. Last error: $lastFailure"
}

function Expand-SafeZip {
    param(
        [Parameter(Mandatory=$true)][string]$Archive,
        [Parameter(Mandatory=$true)][string]$Destination
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $destinationFull = [System.IO.Path]::GetFullPath($Destination)
    $prefix = $destinationFull.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    [System.IO.Directory]::CreateDirectory($destinationFull) | Out-Null
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        foreach ($entry in $zip.Entries) {
            $target = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($destinationFull, $entry.FullName))
            if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "The component archive contains an unsafe path: $($entry.FullName)"
            }
            if ([string]::IsNullOrEmpty($entry.Name)) {
                [System.IO.Directory]::CreateDirectory($target) | Out-Null
                continue
            }
            [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($target)) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
        }
    } finally {
        $zip.Dispose()
    }
}

function Test-OcrRuntime {
    if (-not [System.IO.File]::Exists($runtimePython)) { return $false }
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $runtimePython $probe 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Test-VcRuntime {
    $key = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
    return $null -ne $key -and $key.Installed -eq 1
}

function Install-VcRuntime {
    if (Test-VcRuntime) { return }
    $installer = [System.IO.Path]::Combine($downloadRoot, 'vc_redist.x64.exe')
    Write-Output 'Downloading the Microsoft Visual C++ runtime required by VideoSubFinder...'
    Invoke-WebRequest -UseBasicParsing -Uri $vcRuntimeUrl -OutFile $installer
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or
        $null -eq $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
        throw 'The Microsoft Visual C++ runtime did not have a valid Microsoft signature.'
    }
    Write-Output 'Installing the Microsoft Visual C++ runtime...'
    $process = Start-Process -FilePath $installer -ArgumentList @('/install', '/quiet', '/norestart') -Verb RunAs -Wait -PassThru
    if ($process.ExitCode -notin @(0, 1638, 3010) -or -not (Test-VcRuntime)) {
        throw "The Microsoft Visual C++ runtime installer exited with code $($process.ExitCode)."
    }
}

[System.IO.Directory]::CreateDirectory($componentRoot) | Out-Null
[System.IO.Directory]::CreateDirectory($downloadRoot) | Out-Null
Install-VcRuntime

if (-not [System.IO.File]::Exists($vsfExe)) {
    $vsfArchive = [System.IO.Path]::Combine($downloadRoot, 'VideoSubFinder_6.10_x64.zip')
    Get-VerifiedDownload -Urls $vsfUrls -Destination $vsfArchive -Sha256 $vsfSha256 -Label 'VideoSubFinder 6.10'
    Write-Output 'Installing VideoSubFinder 6.10...'
    $vsfStaging = "$vsfRoot.staging"
    if ([System.IO.Directory]::Exists($vsfStaging)) { Remove-Item -LiteralPath $vsfStaging -Recurse -Force }
    Expand-SafeZip -Archive $vsfArchive -Destination $vsfStaging
    if (-not [System.IO.File]::Exists([System.IO.Path]::Combine($vsfStaging, 'Release_x64', 'VideoSubFinderWXW.exe'))) {
        throw 'The VideoSubFinder archive does not contain the expected Windows executable.'
    }
    if ([System.IO.Directory]::Exists($vsfRoot)) { Remove-Item -LiteralPath $vsfRoot -Recurse -Force }
    Move-Item -LiteralPath $vsfStaging -Destination $vsfRoot
}

if (-not (Test-OcrRuntime)) {
    if (-not [System.IO.File]::Exists($pythonExe)) {
        $pythonInstaller = [System.IO.Path]::Combine($downloadRoot, 'python-3.14.7-amd64.exe')
        Get-VerifiedDownload -Urls @($pythonUrl) -Destination $pythonInstaller -Sha256 $pythonSha256 -Label 'Python 3.14.7 runtime'
        Write-Output 'Installing the private Python runtime...'
        [System.IO.Directory]::CreateDirectory($pythonRoot) | Out-Null
        $arguments = @(
            '/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0',
            'Include_test=0', 'Include_doc=0', 'Include_tcltk=0', 'Shortcuts=0',
            "TargetDir=$pythonRoot"
        )
        $process = Start-Process -FilePath $pythonInstaller -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -ne 0 -or -not [System.IO.File]::Exists($pythonExe)) {
            throw "The private Python runtime installer exited with code $($process.ExitCode)."
        }
    }

    Write-Output 'Creating the OCR environment...'
    if ([System.IO.Directory]::Exists($runtimeRoot)) { Remove-Item -LiteralPath $runtimeRoot -Recurse -Force }
    & $pythonExe -m venv $runtimeRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the OCR virtual environment.' }
    Write-Output 'Installing RapidVideOCR and the CPU inference runtime...'
    & $runtimePython -m pip install --disable-pip-version-check --no-input `
        'rapid_videocr==3.1.1' 'rapidocr==3.9.2' 'onnxruntime==1.29.0'
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned OCR packages.' }
    if (-not (Test-OcrRuntime)) { throw 'The OCR runtime failed its validation probe.' }
}

Write-Output 'SubHooper components are ready.'
