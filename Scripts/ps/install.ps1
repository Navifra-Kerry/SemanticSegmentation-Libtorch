[CmdletBinding()]
Param(
    [Parameter(Mandatory=$false)] [switch] $Execute = $true,
    [Parameter(Mandatory=$false)] [switch] $NoConfirm)

Set-StrictMode -Version latest

Import-Module Download -ErrorAction Stop

$MyDir = Split-Path $MyInvocation.MyCommand.Definition
$ymlDir = Split-Path $MyDir

$roboCopyCmd    = "C:\Windows\System32\robocopy.exe"
$localCache     = "$MyDir\InstallCache"

$3rdPartyPath   = "D:\local"
$OpenCVBasePath = "D:\local\OpenCV"
$LibTorchPath   = "D:\local\LibTorch"  

. "$MyDir\_operations"
. "$MyDir\_verify"
. "$MyDir\_download"
. "$MyDir\_info"
. "$MyDir\_action"

Function main
{
    try {
        if (-not (DisplayStart -NoConfirm $NoConfirm)) {
            Write-Host  
            Write-Host " ... Quitting ... "
            Write-Host
            return
        }
        
        $operations = Set-OperationsInfo 

        $Script:operationList  = @()
        $Script:WinProduct = $null
        if (VerifyOperations -NoConfirm $NoConfirm) 
        {
            DownloadOperations
            
            ActionOperations

            DisplayEnd
        }
    }
    catch {
        Write-Host `nFatal error during script execution!`n($Error[0]).Exception`n
        exit 1
    }
}

main

exit 0
