[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
if ([string]::IsNullOrWhiteSpace($localAppData)) { throw 'Windows LocalAppData could not be resolved.' }
$homeRoot = [IO.Path]::Combine($localAppData, 'SubHooper')
$componentRoot = [IO.Path]::Combine($homeRoot, 'components')
$downloadRoot = [IO.Path]::Combine($homeRoot, 'downloads')
$runtimeRoot = [IO.Path]::Combine($homeRoot, 'runtime')
$pythonRoot = [IO.Path]::Combine($componentRoot, 'Python-3.13.13')
$pythonExe = [IO.Path]::Combine($pythonRoot, 'python.exe')
$cpuRoot = [IO.Path]::Combine($runtimeRoot, 'native-ocr-cpu-py313-v040')
$gpuRoot = [IO.Path]::Combine($runtimeRoot, 'native-ocr-cuda-py313-v040')
$cpuPython = [IO.Path]::Combine($cpuRoot, 'Scripts', 'python.exe')
$gpuPython = [IO.Path]::Combine($gpuRoot, 'Scripts', 'python.exe')
$dmlRoot = [IO.Path]::Combine($runtimeRoot, 'native-ocr-dml-py313-v043')
$dmlPython = [IO.Path]::Combine($dmlRoot, 'Scripts', 'python.exe')
$modelRoot = [IO.Path]::Combine($componentRoot, 'native-ocr-models')
$probe = [IO.Path]::Combine($PSScriptRoot, 'engine', 'probe.py')
$downloadModels = [IO.Path]::Combine($PSScriptRoot, 'engine', 'download_models.py')
$pythonUrl = 'https://www.python.org/ftp/python/3.13.13/python-3.13.13-amd64.exe'
$pythonSha256 = '3c9c81d80f91c002ced86d645422d81432c68c7d9b6b0e974768ca2e449a4d00'
$opencvWheel = [IO.Path]::Combine($downloadRoot, 'opencv_python_headless-4.14.0.94-cp37-abi3-win_amd64.whl')
$opencvUrl = 'https://files.pythonhosted.org/packages/ad/8d/db8673846ee53cbb5de4c2b4decc11cf733e203eb7d5146297869f69bd48/opencv_python_headless-4.14.0.94-cp37-abi3-win_amd64.whl'
$opencvSha256 = 'cbed65415b8f6a9541c705afe3e64795840524d0ff3bc58f507826284a1dc64b'
$pythonMaxBytes = 128MB
$opencvMaxBytes = 256MB
function Invoke-ComponentDownload {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][long]$MaximumBytes
    )

    $response = $null
    $inputStream = $null
    $outputStream = $null
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $requestUri = [Uri]$Url
        for ($redirectCount = 0; $redirectCount -le 10; $redirectCount++) {
            if (-not $requestUri.IsAbsoluteUri -or $requestUri.Scheme -ne 'https') {
                throw [IO.InvalidDataException]::new('Component downloads require HTTPS, including redirects.')
            }
            $request = [Net.HttpWebRequest]::Create($requestUri)
            $request.UserAgent = 'SubHooper/0.4.3'
            $request.Timeout = 30000
            $request.ReadWriteTimeout = 60000
            $request.AllowAutoRedirect = $false
            $response = $request.GetResponse()
            if ([int]$response.StatusCode -notin @(301, 302, 303, 307, 308)) { break }
            $location = $response.Headers['Location']
            if ([string]::IsNullOrWhiteSpace($location) -or $redirectCount -eq 10) {
                throw [IO.InvalidDataException]::new('Component download redirect is invalid or exceeds the limit.')
            }
            $nextUri = [Uri]::new($requestUri, $location)
            $response.Dispose()
            $response = $null
            $requestUri = $nextUri
        }
        if ($response.ContentLength -gt $MaximumBytes) {
            throw [IO.InvalidDataException]::new('Download size exceeds the configured artifact limit.')
        }
        $inputStream = $response.GetResponseStream()
        $outputStream = [IO.File]::Open($Destination, [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write, [IO.FileShare]::None)
        $buffer = New-Object byte[] (256 * 1024)
        $received = [long]0
        $nextUpdate = [long](1024 * 1024)
        while (($count = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            if ($received + $count -gt $MaximumBytes) {
                throw [IO.InvalidDataException]::new('Download size exceeds the configured artifact limit.')
            }
            $outputStream.Write($buffer, 0, $count)
            $received += $count
            if ($received -ge $nextUpdate) {
                $mb = [Math]::Round($received / 1MB, 1)
                if ($response.ContentLength -gt 0) {
                    $percent = [Math]::Min(100, [Math]::Round(100 * $received / $response.ContentLength))
                    Write-Output "Component download: $percent% ($mb MB)"
                } else {
                    Write-Output "Component download: $mb MB received"
                }
                $nextUpdate = $received + 1MB
            }
        }
        Write-Output "Component download complete: $([Math]::Round($received / 1MB, 1)) MB"
        return
    } catch {
        if ($_.Exception -is [IO.InvalidDataException]) { throw }
        # Retain the curl fallback for hosts where Windows .NET proxy/TLS setup fails.
        $failure = $_.Exception.Message
        if ($outputStream) { $outputStream.Dispose(); $outputStream = $null }
        if ($inputStream) { $inputStream.Dispose(); $inputStream = $null }
        if ($response) { $response.Dispose(); $response = $null }
        Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
        $curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
        if (-not $curl) { throw "Component download failed: $failure" }
        $curlVersionText = (& $curl.Source --version | Select-Object -First 1)
        if ($curlVersionText -notmatch '^curl (\d+\.\d+\.\d+)') {
            throw [IO.InvalidDataException]::new('Could not verify curl download size-limit support.')
        }
        if ([version]$Matches[1] -lt [version]'8.4.0') {
            throw [IO.InvalidDataException]::new('Safe download fallback requires curl 8.4 or newer. Update Windows and retry.')
        }
        Write-Output "Retrying component download with curl.exe: $failure"
        & $curl.Source --silent --show-error --location --fail --max-filesize $MaximumBytes `
            --proto '=https' --proto-redir '=https' --max-redirs 10 `
            --retry 3 --retry-delay 2 --connect-timeout 30 --max-time 600 `
            --speed-time 30 --speed-limit 1024 --output $Destination $Url
        if ($LASTEXITCODE -eq 63) { throw [IO.InvalidDataException]::new('Downloaded file exceeds the configured artifact limit.') }
        if ($LASTEXITCODE -ne 0) { throw "curl.exe exited with code $LASTEXITCODE." }
        if ([long](Get-Item -LiteralPath $Destination).Length -gt $MaximumBytes) {
            throw [IO.InvalidDataException]::new('Downloaded file exceeds the configured artifact limit.')
        }
    } finally {
        if ($outputStream) { $outputStream.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        if ($response) { $response.Dispose() }
    }
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory=$true)][string[]]$Urls,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][string]$Sha256,
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][long]$MaximumBytes
    )
    $expected = $Sha256.ToLowerInvariant()
    if ([System.IO.File]::Exists($Destination)) {
        $existing = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existing -eq $expected) { return }
        Remove-Item -LiteralPath $Destination -Force
    }

    $lastFailure = 'No download attempt completed.'
    for ($sourceIndex = 0; $sourceIndex -lt $Urls.Count; $sourceIndex++) {
        $url = $Urls[$sourceIndex]
        foreach ($attempt in 1..3) {
            $partial = "$Destination.$([guid]::NewGuid().ToString('N')).partial"
            Write-Output "Downloading $Label (source $($sourceIndex + 1)/$($Urls.Count), attempt $attempt/3)..."
            try {
                Invoke-ComponentDownload -Url $url -Destination $partial -MaximumBytes $MaximumBytes
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
                if ($_.Exception -is [IO.InvalidDataException]) { throw }
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


function Remove-LegacyOcrComponents([string]$ManagedHome) {
    # Only exact obsolete SubHooper paths with recognizable component markers.
    # Refuse junctions/symlinks in ancestors or descendants before recursive removal.
    $base = [IO.Path]::GetFullPath($ManagedHome).TrimEnd('\', '/')
    $entries = @(
        @('components\VideoSubFinder-6.10', 'Release_x64\VideoSubFinderWXW.exe'),
        @('runtime\ocr-cpu-py314-auto', 'Lib\site-packages\rapid_videocr'),
        @('runtime\ocr-gpu-py314-auto', 'Lib\site-packages\rapid_videocr')
    )
    foreach ($entry in $entries) {
        try {
            $target = [IO.Path]::GetFullPath([IO.Path]::Combine($base, $entry[0]))
            if (-not $target.StartsWith($base + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Legacy component path escaped the managed directory.'
            }
            if (-not (Test-Path -LiteralPath $target)) { continue }
            $cursor = $target
            while ($cursor) {
                $item = Get-Item -LiteralPath $cursor -Force
                if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                    throw 'Legacy component path contains a junction or symbolic link.'
                }
                $cursor = [IO.Path]::GetDirectoryName($cursor)
            }
            if (-not (Test-Path -LiteralPath ([IO.Path]::Combine($target, $entry[1])))) {
                throw 'Legacy component marker was not found.'
            }
            $pending = New-Object 'System.Collections.Generic.Stack[string]'
            $pending.Push($target)
            while ($pending.Count -gt 0) {
                foreach ($child in (Get-ChildItem -LiteralPath $pending.Pop() -Force)) {
                    if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                        throw 'Legacy component contains a junction or symbolic link.'
                    }
                    if ($child.PSIsContainer) { $pending.Push($child.FullName) }
                }
            }
            Remove-Item -LiteralPath $target -Recurse -Force
            Write-Output "Removed obsolete OCR component: $($entry[0])"
        } catch {
            Write-Output "Legacy OCR cleanup skipped $($entry[0]): $($_.Exception.Message)"
        }
    }
}

function Test-NativeRuntime([string]$Executable) {
    if (-not [IO.File]::Exists($Executable)) { return $false }
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable $probe $modelRoot 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

function Install-VerifiedOpenCv([string]$Executable) {
    Get-VerifiedDownload -Urls @($opencvUrl) -Destination $opencvWheel -Sha256 $opencvSha256 `
        -Label 'OpenCV 4.14.0.94 Windows wheel' -MaximumBytes $opencvMaxBytes
    & $Executable -m pip install --disable-pip-version-check --no-input --no-deps $opencvWheel
    if ($LASTEXITCODE -ne 0) { throw 'The verified OpenCV wheel could not be installed.' }
}

[IO.Directory]::CreateDirectory($componentRoot) | Out-Null
[IO.Directory]::CreateDirectory($downloadRoot) | Out-Null
[IO.Directory]::CreateDirectory($runtimeRoot) | Out-Null
if (-not [IO.File]::Exists($pythonExe)) {
    $installer = [IO.Path]::Combine($downloadRoot, 'python-3.13.13-amd64.exe')
    Get-VerifiedDownload -Urls @($pythonUrl) -Destination $installer -Sha256 $pythonSha256 `
        -Label 'Python 3.13.13 runtime' -MaximumBytes $pythonMaxBytes
    [IO.Directory]::CreateDirectory($pythonRoot) | Out-Null
    $arguments = @('/quiet','InstallAllUsers=0','PrependPath=0','Include_launcher=0',
        'Include_test=0','Include_doc=0','Include_tcltk=0','Shortcuts=0',"TargetDir=`"$pythonRoot`"")
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -ne 0 -or -not [IO.File]::Exists($pythonExe)) {
        throw "The private Python runtime installer exited with code $($process.ExitCode)."
    }
}
# The downloader verifies each version-pinned ONNX asset and dictionary by SHA-256.
& $pythonExe $downloadModels $modelRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not download and verify the OCR model assets.' }
if (-not (Test-NativeRuntime $cpuPython)) {
    if (Test-Path -LiteralPath $cpuRoot) { Remove-Item -LiteralPath $cpuRoot -Recurse -Force }
    & $pythonExe -m venv $cpuRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the OCR CPU environment.' }
    Write-Output 'Installing pinned native OCR CPU packages...'
    & $cpuPython -m pip install --disable-pip-version-check --no-input `
        'numpy==2.2.6' 'onnxruntime==1.22.1'
    if ($LASTEXITCODE -ne 0) { throw 'The OCR CPU packages could not be installed.' }
    Install-VerifiedOpenCv $cpuPython
    if (-not (Test-NativeRuntime $cpuPython)) {
        throw 'The OCR CPU environment failed validation.'
    }
}
# CUDA is optional. The verified CPU path remains usable if installation fails.
$cudaDevice = $false
if (Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue) {
    $previousGpuPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $deviceNames = & nvidia-smi.exe --query-gpu=name --format=csv,noheader 2>$null
        $cudaDevice = $LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($deviceNames | Out-String))
    } finally { $ErrorActionPreference = $previousGpuPreference }
}
if ($cudaDevice) {
    if (-not (Test-NativeRuntime $gpuPython)) {
        try {
            if (Test-Path -LiteralPath $gpuRoot) { Remove-Item -LiteralPath $gpuRoot -Recurse -Force }
            & $pythonExe -m venv $gpuRoot
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the OCR CUDA environment.' }
            Write-Output 'Installing pinned native OCR CUDA packages...'
            & $gpuPython -m pip install --disable-pip-version-check --no-input `
                'numpy==2.2.6' 'onnxruntime-gpu[cuda,cudnn]==1.22.0' `
                'nvidia-cuda-runtime-cu12==12.9.79' `
                'nvidia-cudnn-cu12==9.26.0.51' `
                'nvidia-cublas-cu12==12.9.1.4' `
                'nvidia-cuda-nvrtc-cu12==12.9.86' `
                'nvidia-cufft-cu12==11.4.1.4' `
                'nvidia-curand-cu12==10.3.10.19' `
                'nvidia-nvjitlink-cu12==12.9.86'
            if ($LASTEXITCODE -ne 0) { throw 'The OCR CUDA packages could not be installed.' }
            Install-VerifiedOpenCv $gpuPython
            if (-not (Test-NativeRuntime $gpuPython)) {
                throw 'The OCR CUDA environment failed validation.'
            }
        } catch {
            Write-Output "CUDA setup unavailable; CPU OCR is ready. $($_.Exception.Message)"
        }
    }
}
# DirectML is isolated from CUDA and CPU wheels, which share the onnxruntime namespace.
# NVIDIA systems keep the validated CUDA runtime; other PCs try DirectX 12 GPU OCR.
if (-not ($cudaDevice -and (Test-NativeRuntime $gpuPython)) -and -not (Test-NativeRuntime $dmlPython)) {
    try {
        if (Test-Path -LiteralPath $dmlRoot) { Remove-Item -LiteralPath $dmlRoot -Recurse -Force }
        & $pythonExe -m venv $dmlRoot
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the OCR DirectML environment.' }
        Write-Output 'Installing pinned native OCR DirectML packages...'
        & $dmlPython -m pip install --disable-pip-version-check --no-input `
            'numpy==2.2.6' 'onnxruntime-directml==1.22.0'
        if ($LASTEXITCODE -ne 0) { throw 'The OCR DirectML packages could not be installed.' }
        Install-VerifiedOpenCv $dmlPython
        if (-not (Test-NativeRuntime $dmlPython)) {
            throw 'The OCR models could not run on a DirectX 12 hardware adapter.'
        }
    } catch {
        Write-Output "DirectML setup unavailable; CPU OCR is ready. $($_.Exception.Message)"
    }
}
if (Test-NativeRuntime $cpuPython) {
    Remove-LegacyOcrComponents $homeRoot
} else {
    throw 'Native OCR validation failed; legacy components were retained.'
}
Write-Output 'SubHooper native OCR components are ready.'
