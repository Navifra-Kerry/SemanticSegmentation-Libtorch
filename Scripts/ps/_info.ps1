function FunctionIntro(
    [Parameter(Mandatory = $true)][hashtable] $table) {
    $table | Out-String | Write-Verbose
}

function GetKey(
    [string] $validChar) {
    do {
        $key = Read-Host
    } until ($key -match $validChar)

    return $key
}

function DisplayStartMessage {
"
This script sets up Exampe prerequisites and Exampe environment on your system.
The script will analyse your machine and will determine which components are required. 
The required components will be downloaded in [$localCache]
Repeated operation of this script will reuse already downloaded components.
    - LibTorch1.3 will be installed into [$LibTorchPath]
    - OpenCV4.1.1 will be installed into [$OpenCVBasePath]
 "
}

function Display64BitWarningMessage {
    "
A 64bit version of Powershell is required to run this script.
Please make sure you started this installation from a 64bit command process.
"
}

function DisplayVersionWarningMessage(
    [string] $version) {
    "
You are executing this script from Powershell Version $version.
We recommend that you execute the script from Powershell Version 4 or later. You can install Powershell Version 4 from:
    https://www.microsoft.com/en-us/download/details.aspx?id=40855
"
}

function DisplayWarningNoExecuteMessage {
    "
The parameter '-Execute:$false' has been supplied to the script.
The script will execute without making any actual changes to the machine.
"
}

function DisplayStartContinueMessage {
    "
1 - I agree and want to continue
Q - Quit the installation process
"
}

function CheckPowershellVersion {
    $psVersion = $PSVersionTable.PSVersion.Major
    if ($psVersion -ge 4) {
        return $true
    }

    Write-Host $(DisplayVersionWarningMessage $psVersion)
    if ($psVersion -eq 3) {
        return $true
    }
    return $false
}

function CheckOSVersion {
    $runningOn = (Get-WmiObject -class Win32_OperatingSystem).Caption
    $isMatching = ($runningOn -match "^Microsoft Windows (8\.1|10|Server 2012 R2|Server 2016)") 
    if ($isMatching) {
        return
    }

    Write-Warning "
You are running this script on [$runningOn].
The Microsoft Cognitive Toolkit is designed and tested on Windows 8.1, Windows 10, 
Windows Server 2012 R2, and Windows Server 2016.
"
    return
}

function Check64BitProcess {
    if ([System.Environment]::Is64BitProcess) {
        return $true
    }

    Write-Warning $(Display64BitWarningMessage)
    return $false
}

function DisplayStart(
    [bool] $NoConfirm) {
    Write-Host $(DisplayStartMessage)
    if (-not (Check64BitProcess)) {
        return $false
    }
    if (-not (CheckPowershellVersion)) {
        return $false
    }

    CheckOSVersion

    if (-not $Execute) {
        Write-Warning $(DisplayWarningNoExecuteMessage)
    }
    if ($NoConfirm) {
        return $true
    }
    Write-Host $(DisplayStartContinueMessage)
    $choice = GetKey '^[1qQ]+$'

    if ($choice -contains "1") {
        return $true
    }

    return $false
}

Function DisplayEnd() {
    if (-not $Execute) { return }

    Write-Host "

Transfer Learning Dependency install complete.
"
}
