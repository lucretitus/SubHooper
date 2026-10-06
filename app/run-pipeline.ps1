[CmdletBinding()]
param(
    [Parameter(Mandatory=$true,Position=0)][string]$Video,
    [string]$PythonPath = $env:SUBTITLE_PYTHON,
    [ValidateSet('Native')][string]$Engine = 'Native',
    [ValidateSet('Bottom','LowerHalf','Full','Custom')][string]$SubtitleRegion = 'Bottom',
    [ValidateRange(0.0,1.0)][double]$RegionTop = 0.42,
    [ValidateRange(0.0,1.0)][double]$RegionBottom = 0.02,
    [ValidateRange(0.0,1.0)][double]$RegionLeft = 0.03,
    [ValidateRange(0.0,1.0)][double]$RegionRight = 0.97,
    [ValidateSet('Auto','CUDA','CPU')][string]$Compute = 'Auto',
    [ValidateSet('Auto','CUDA','CPU','Mixed','DML')][string]$OCRCompute = 'Auto',
    [ValidateSet('CLI','GUI')][string]$Client = 'CLI',
    [ValidateRange(1,86400)][int]$TimeoutSeconds = 14400,
    [switch]$CollectDiagnostics,
    [switch]$KeepTemp
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$script:runtimeErrors = @{}
function Test-NativeRuntime([string]$Python, [string]$Probe, [string]$Models) {
    if (-not [IO.File]::Exists($Python)) {
        $script:runtimeErrors[$Python] = "OCR runtime is not installed: $Python"
        return $false
    }
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $probeOutput = @(& $Python $Probe $Models 2>&1)
        if ($LASTEXITCODE -eq 0) { return $true }
        $script:runtimeErrors[$Python] = ($probeOutput | ForEach-Object { [string]$_ }) -join [Environment]::NewLine
        return $false
    } catch {
        $script:runtimeErrors[$Python] = $_.Exception.Message
        return $false
    }
    finally { $ErrorActionPreference = $previous }
}
try {
    $inputVideo = (Get-Item -LiteralPath $Video -ErrorAction Stop).FullName
    $localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    $homeRoot = [IO.Path]::Combine($localAppData, 'SubHooper')
    $cpuPython = [IO.Path]::Combine($homeRoot,'runtime','native-ocr-cpu-py313-v040','Scripts','python.exe')
    $gpuPython = [IO.Path]::Combine($homeRoot,'runtime','native-ocr-cuda-py313-v040','Scripts','python.exe')
    $dmlPython = [IO.Path]::Combine($homeRoot,'runtime','native-ocr-dml-py313-v043','Scripts','python.exe')
    $modelRoot = [IO.Path]::Combine($homeRoot,'components','native-ocr-models')
    $probe = [IO.Path]::Combine($PSScriptRoot,'engine','probe.py')
    $selected = if ($PythonPath) { $PythonPath } else { $cpuPython }
    if (-not (Test-NativeRuntime $selected $probe $modelRoot)) {
        throw 'Native OCR models or CPU runtime are invalid. Open Settings > Components.'
    }
    $ocrPython = $selected
    $ocrMode = 'cpu'
    $cudaDevice = $false
    if ($OCRCompute -notin @('CPU','DML') -and (Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue)) {
        $previousGpuPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $deviceNames = & nvidia-smi.exe --query-gpu=name --format=csv,noheader 2>$null
            $cudaDevice = $LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($deviceNames | Out-String))
        } finally { $ErrorActionPreference = $previousGpuPreference }
    }
    if ($OCRCompute -ne 'CPU') {
        if ($cudaDevice -and (Test-NativeRuntime $gpuPython $probe $modelRoot)) {
                $ocrPython = $gpuPython
                $ocrMode = if ($OCRCompute -eq 'CUDA') { 'cuda' } else { 'mixed' }
                $sitePackages = [IO.Path]::Combine((Split-Path (Split-Path $gpuPython -Parent) -Parent),'Lib','site-packages')
                $dllPaths = @([IO.Path]::Combine($sitePackages,'nvidia','cudnn','bin'),
                              [IO.Path]::Combine($sitePackages,'nvidia','cuda_runtime','bin')) |
                    Where-Object { Test-Path -LiteralPath $_ -PathType Container }
                if ($dllPaths.Count -gt 0) { $env:PATH = (($dllPaths -join ';') + ';' + $env:PATH) }
        }
        if ($ocrMode -eq 'cpu' -and $OCRCompute -ne 'CUDA' -and
            (Test-NativeRuntime $dmlPython $probe $modelRoot)) {
            $ocrPython = $dmlPython
            $ocrMode = 'dml'
        }
        if (($OCRCompute -eq 'CUDA' -and $ocrMode -ne 'cuda') -or
            ($OCRCompute -eq 'DML' -and $ocrMode -ne 'dml') -or
            ($OCRCompute -eq 'Mixed' -and $ocrMode -notin @('mixed','dml'))) {
            $details = @()
            if (-not $cudaDevice -and $OCRCompute -ne 'DML') {
                $details += 'NVIDIA GPU detection failed; check the NVIDIA driver.'
            }
            foreach ($runtime in @($gpuPython, $dmlPython)) {
                if ($script:runtimeErrors.ContainsKey($runtime)) { $details += $script:runtimeErrors[$runtime] }
            }
            throw ('GPU OCR is unavailable. Open Settings > Video Extraction Components > Verify Components, then retry. ' + ($details -join [Environment]::NewLine))
        }
    }
    $region = $SubtitleRegion.ToLowerInvariant().Replace('lowerhalf', 'lower-half')
    $argsList = @([IO.Path]::Combine($PSScriptRoot,'engine','pipeline.py'),$inputVideo,
        '--region',$region,'--compute',$Compute.ToLowerInvariant(),
        '--ocr-requested',$OCRCompute.ToLowerInvariant(),'--ocr-compute',$ocrMode,
        '--ocr-python',$ocrPython,'--models-dir',$modelRoot,
        '--client',$Client.ToLowerInvariant(),'--timeout',"$TimeoutSeconds")
    if ($SubtitleRegion -eq 'Custom') {
        $invariant = [Globalization.CultureInfo]::InvariantCulture
        $argsList += @('--region-top',$RegionTop.ToString('0.######',$invariant),
            '--region-bottom',$RegionBottom.ToString('0.######',$invariant),
            '--region-left',$RegionLeft.ToString('0.######',$invariant),
            '--region-right',$RegionRight.ToString('0.######',$invariant))
    }
    if ($CollectDiagnostics) { $argsList += '--collect-diagnostics' }
    if ($KeepTemp) { $argsList += '--keep-temp' }
    & $selected @argsList
    exit $LASTEXITCODE
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
