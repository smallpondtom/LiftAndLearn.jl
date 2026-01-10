@REM Usage
@REM Change directory to scripts/Streaming-OpInf then run:
@REM run_experiments.bat burgers
@REM run_experiments.bat kse
@REM run_experiments.bat both

@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Error: No experiment specified.
    goto :usage
)

if /i "%1"=="burgers" goto :run_burgers
if /i "%1"=="kse" goto :run_kse
if /i "%1"=="both" goto :run_both

echo Error: Unknown experiment '%1'
goto :usage

:run_burgers
echo Running Burgers experiment...
cd /d "%~dp0\burgers"
julia --project=../.. main.jl
if !errorlevel! neq 0 (
    echo Burgers experiment failed!
    cd /d "%~dp0"
    exit /b 1
)
cd /d "%~dp0"
echo Burgers experiment completed.
goto :end

:run_kse
echo Running KSE experiment...
cd /d "%~dp0\kse"
julia --project=../.. main.jl
if !errorlevel! neq 0 (
    echo KSE experiment failed!
    cd /d "%~dp0"
    exit /b 1
)
cd /d "%~dp0"
echo KSE experiment completed.
goto :end

:run_both
echo Running both experiments...
call :run_burgers_internal
if !errorlevel! neq 0 exit /b 1
echo.
call :run_kse_internal
if !errorlevel! neq 0 exit /b 1
echo All experiments completed successfully!
goto :end

:run_burgers_internal
echo Running Burgers experiment...
cd /d "%~dp0\burgers"
julia --project=.. main.jl
cd /d "%~dp0"
if !errorlevel! neq 0 (
    echo Burgers experiment failed!
    exit /b 1
)
echo Burgers experiment completed.
exit /b 0

:run_kse_internal
echo Running KSE experiment...
cd /d "%~dp0\kse"
julia --project=.. main.jl
cd /d "%~dp0"
if !errorlevel! neq 0 (
    echo KSE experiment failed!
    exit /b 1
)
echo KSE experiment completed.
exit /b 0

:usage
echo.
echo Usage: %0 [burgers^|kse^|both]
echo   burgers - Run only the Burgers experiment
echo   kse     - Run only the KSE experiment
echo   both    - Run both experiments
echo.
echo Examples:
echo   %0 burgers
echo   %0 kse
echo   %0 both
exit /b 1

:end
echo Experiment(s) finished successfully!
exit /b 0
