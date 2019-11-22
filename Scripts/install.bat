@echo Transfer Learning Pytorch Binary Install Script
@echo off

setlocal
set allArgs=%*
set installBatchDir=%~dp0
set psInstall=%installBatchDir%ps\install.ps1
set PSModulePath=%PSModulePath%;%installBatchDir%ps\Modules

:loop
if /I "%~1"=="-h" goto :helpMsg
if /I "%~1"=="/h" goto :helpMsg
if /I "%~1"=="-help" goto :helpMsg
if /I "%~1"=="/help" goto :helpMsg
if "%~1"=="-?" goto :helpMsg
if "%~1"=="/?" goto :helpMsg
shift

if not "%~1"=="" goto loop

powershell -NoProfile -NoLogo -ExecutionPolicy Bypass %psInstall% %allArgs%
if errorlevel 1 (
  @echo Error during install operation
  exit /b 1
)
goto :eof

:helpMsg
@echo.
@echo More help can be Contect Me: 
@echo     cdh0012@naver.com
@echo.

