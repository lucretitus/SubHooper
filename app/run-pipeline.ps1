[CmdletBinding()]
param(
    [Parameter(Mandatory=$true,Position=0)][string]$Video,
    [string]$PythonPath = $env:SUBTITLE_PYTHON,
    [string]$VsfPath = $env:SUBTITLE_VSF,
    [ValidateSet('Bottom','LowerHalf','Full','Custom')][string]$SubtitleRegion = 'Bottom',
    [ValidateRange(0.0,1.0)][double]$RegionTop = 0.42,
    [ValidateRange(0.0,1.0)][double]$RegionBottom = 0.02,
    [ValidateRange(0.0,1.0)][double]$RegionLeft = 0.03,
    [ValidateRange(0.0,1.0)][double]$RegionRight = 0.97,
    [ValidateSet('Auto','CUDA','CPU')][string]$Compute = 'Auto',
    [ValidateSet('Auto','CUDA','CPU')][string]$OCRCompute = 'Auto',
    [ValidateSet('CLI','GUI')][string]$Client = 'CLI',
    [ValidateRange(1,86400)][int]$TimeoutSeconds = 14400,
    [switch]$CollectDiagnostics,
    [switch]$KeepTemp
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
if ([string]::IsNullOrWhiteSpace($localAppData)) {
    throw 'Windows LocalAppData could not be resolved.'
}
$resultsRoot = if ($env:SUBTITLE_RESULTS_ROOT) {
    [System.IO.Path]::GetFullPath($env:SUBTITLE_RESULTS_ROOT)
} else {
    [System.IO.Path]::Combine($PSScriptRoot, 'results')
}
[System.IO.Directory]::CreateDirectory($resultsRoot) | Out-Null
$reportsRoot = if ($env:SUBTITLE_REPORTS_ROOT) {
    [System.IO.Path]::GetFullPath($env:SUBTITLE_REPORTS_ROOT)
} else {
    [System.IO.Path]::Combine($resultsRoot, 'reports')
}
[System.IO.Directory]::CreateDirectory($reportsRoot) | Out-Null

function Test-CompatibleOcrPython {
    param(
        [Parameter(Mandatory=$true)][string]$Candidate,
        [Parameter(Mandatory=$true)][string]$Probe
    )
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $false }
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Candidate $Probe 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Test-BasePython {
    param([Parameter(Mandatory=$true)][string]$Candidate)
    if (-not [System.IO.File]::Exists($Candidate)) { return $false }
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $version = & $Candidate -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))' 2>$null
        return $LASTEXITCODE -eq 0 -and (($version | Select-Object -Last 1) -eq '3.14')
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Get-BasePythonCandidates {
    param([Parameter(Mandatory=$true)][string]$LocalAppData)
    $found = @()
    $pythonRoot = [System.IO.Path]::Combine($LocalAppData, 'Python')
    $found += [System.IO.Path]::Combine($pythonRoot, 'pythoncore-3.14-64', 'python.exe')
    if ([System.IO.Directory]::Exists($pythonRoot)) {
        foreach ($directory in [System.IO.Directory]::EnumerateDirectories($pythonRoot)) {
            $found += [System.IO.Path]::Combine($directory, 'python.exe')
        }
    }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $found += $pythonCommand.Source }
    $pyCommand = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyCommand) {
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $reported = & $pyCommand.Source -3.14 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $found += ($reported | Select-Object -Last 1) }
        } finally {
            $ErrorActionPreference = $previousPreference
        }
    }
    return @($found | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
}

function Initialize-StableCpuRuntime {
    param(
        [Parameter(Mandatory=$true)][string]$SourcePython,
        [Parameter(Mandatory=$true)][string]$TargetPython,
        [Parameter(Mandatory=$true)][string]$Probe,
        [Parameter(Mandatory=$true)][string]$SetupLog
    )
    if (Test-CompatibleOcrPython -Candidate $TargetPython -Probe $Probe) { return $TargetPython }
    $runtimeRoot = Split-Path (Split-Path $TargetPython -Parent) -Parent
    $runtimeParent = Split-Path $runtimeRoot -Parent
    Write-Host 'Preparing the shared OCR CPU runtime...'
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if (Test-Path -LiteralPath $runtimeRoot -PathType Container) {
            Remove-Item -LiteralPath $runtimeRoot -Recurse -Force
        }
        New-Item -ItemType Directory -Path $runtimeParent -Force | Out-Null
        & $SourcePython -m venv $runtimeRoot 2>&1 |
            Set-Content -LiteralPath $SetupLog -Encoding UTF8
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the OCR CPU virtual environment.' }
        & $TargetPython -m pip install --disable-pip-version-check --no-input `
            'rapid_videocr==3.1.1' 'rapidocr==3.9.2' 'onnxruntime==1.29.0' 2>&1 |
            Add-Content -LiteralPath $SetupLog -Encoding UTF8
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the OCR CPU packages.' }
        if (Test-CompatibleOcrPython -Candidate $TargetPython -Probe $Probe) {
            Write-Host 'The shared OCR CPU runtime is ready.'
            return $TargetPython
        }
        throw 'The prepared OCR CPU runtime failed version validation.'
    } catch {
        throw
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

try {
    $inputVideo = (Get-Item -LiteralPath $Video).FullName
    $stableCpuRuntime = [System.IO.Path]::Combine(
        $localAppData, 'SubHooper', 'runtime', 'ocr-cpu-py314-auto')
    $stableCpuPython = [System.IO.Path]::Combine($stableCpuRuntime, 'Scripts', 'python.exe')
    $legacyCpuPython = [System.IO.Path]::Combine(
        $localAppData, 'SubtitleExtractor', 'runtime', 'ocr-cpu-py314-auto', 'Scripts', 'python.exe')
    $probe = [System.IO.Path]::Combine($PSScriptRoot, 'engine', 'probe.py')
    $selected = $null
    if ($PythonPath) {
        if (-not (Test-CompatibleOcrPython -Candidate $PythonPath -Probe $probe)) {
            throw 'SUBTITLE_PYTHON does not point to a compatible OCR runtime.'
        }
        $selected = $PythonPath
    } elseif (Test-CompatibleOcrPython -Candidate $stableCpuPython -Probe $probe) {
        $selected = $stableCpuPython
    } elseif (Test-CompatibleOcrPython -Candidate $legacyCpuPython -Probe $probe) {
        $selected = $legacyCpuPython
    } else {
        $discoveryLog = [System.IO.Path]::Combine($reportsRoot, 'python-discovery-latest.log')
        $discoveryLines = @(
            "Generated=$([DateTimeOffset]::Now.ToString('o'))",
            "LocalAppData=$localAppData",
            "StableCpuPython=$stableCpuPython"
        )
        $basePython = $null
        foreach ($candidate in @(Get-BasePythonCandidates -LocalAppData $localAppData)) {
            $exists = [System.IO.File]::Exists($candidate)
            $compatible = $false
            if ($exists) {
                $compatible = Test-BasePython -Candidate $candidate
            }
            $discoveryLines += "Candidate=$candidate|Exists=$exists|Python314=$compatible"
            if (-not $basePython -and $compatible) {
                $basePython = $candidate
            }
        }
        $discoveryLines += "Selected=$basePython"
        [System.IO.File]::WriteAllLines(
            $discoveryLog,
            $discoveryLines,
            [System.Text.UTF8Encoding]::new($false))
        if (-not $basePython) {
            throw "The managed OCR runtime is not installed. Open Settings > Components in SubHooper. Log: $discoveryLog"
        }
        $selected = Initialize-StableCpuRuntime `
            -SourcePython $basePython `
            -TargetPython $stableCpuPython `
            -Probe $probe `
            -SetupLog ([System.IO.Path]::Combine($reportsRoot, 'ocr-cpu-setup-latest.log'))
    }
    Write-Host 'OCR CPU runtime validated.'

    $ocrSelected = 'cpu'
    $ocrPython = $selected
    if ($OCRCompute -ne 'CPU') {
        $gpuRuntime = [System.IO.Path]::Combine(
            $localAppData, 'SubHooper', 'runtime', 'ocr-gpu-py314-auto')
        $gpuPython = [System.IO.Path]::Combine($gpuRuntime, 'Scripts', 'python.exe')
        $legacyGpuPython = [System.IO.Path]::Combine(
            $localAppData, 'SubtitleExtractor', 'runtime', 'ocr-gpu-py314-auto', 'Scripts', 'python.exe')
        if (-not (Test-Path -LiteralPath $gpuPython -PathType Leaf) -and
            (Test-Path -LiteralPath $legacyGpuPython -PathType Leaf)) {
            $gpuPython = $legacyGpuPython
        }
        if (Test-Path -LiteralPath $gpuPython -PathType Leaf) {
            $probeLog = [System.IO.Path]::Combine($reportsRoot, 'ocr-gpu-probe-latest.log')
            $previousErrorAction = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                & $gpuPython ([System.IO.Path]::Combine($PSScriptRoot, 'engine', 'ocr_gpu_probe.py')) 2>&1 |
                    Set-Content -LiteralPath $probeLog -Encoding UTF8
                $probeCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorAction
            }
            if ($probeCode -eq 0) {
                $ocrSelected = 'cuda'
                $ocrPython = $gpuPython
            }
        }
        if ($OCRCompute -eq 'CUDA' -and $ocrSelected -ne 'cuda') {
            throw 'The optional OCR CUDA runtime is unavailable. Select Auto or CPU in SubHooper.'
        }
    }
    $region = $SubtitleRegion.ToLowerInvariant().Replace('lowerhalf', 'lower-half')
    $arguments = @(
        ([System.IO.Path]::Combine($PSScriptRoot, 'engine', 'pipeline.py')), $inputVideo,
        '--region', $region, '--compute', $Compute.ToLowerInvariant(),
        '--ocr-requested', $OCRCompute.ToLowerInvariant(),
        '--ocr-compute', $ocrSelected, '--ocr-python', $ocrPython,
        '--client', $Client.ToLowerInvariant(),
        '--timeout', "$TimeoutSeconds"
    )
    if ($SubtitleRegion -eq 'Custom') {
        $invariant = [System.Globalization.CultureInfo]::InvariantCulture
        $arguments += @(
            '--region-top', $RegionTop.ToString('0.######', $invariant),
            '--region-bottom', $RegionBottom.ToString('0.######', $invariant),
            '--region-left', $RegionLeft.ToString('0.######', $invariant),
            '--region-right', $RegionRight.ToString('0.######', $invariant)
        )
    }
    if ($VsfPath) { $arguments += @('--vsf', $VsfPath) }
    if ($CollectDiagnostics) { $arguments += '--collect-diagnostics' }
    if ($KeepTemp) { $arguments += '--keep-temp' }
    & $selected @arguments
    $pipelineCode = $LASTEXITCODE
    try { Get-Content -LiteralPath ([System.IO.Path]::Combine($resultsRoot, 'latest-summary.txt')) -Raw -Encoding UTF8 | Set-Clipboard } catch { }
    exit $pipelineCode
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

