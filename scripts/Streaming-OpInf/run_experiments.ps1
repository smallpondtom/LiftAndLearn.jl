# Usage
# Change directory to scripts/Streaming-OpInf then run:
# .\run_experiments.ps1 -Experiment burgers
# .\run_experiments.ps1 -Experiment kse  
# .\run_experiments.ps1 -Experiment both

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("burgers", "kse", "both")]
    [string]$Experiment
)

# Function to run Burgers experiment
function Run-Burgers {
    Write-Host "Running Burgers experiment..." -ForegroundColor Green
    Push-Location burgers
    try {
        julia --project=../.. main.jl
        Write-Host "Burgers experiment completed." -ForegroundColor Green
    }
    catch {
        Write-Host "Error running Burgers experiment: $_" -ForegroundColor Red
        return $false
    }
    finally {
        Pop-Location
    }
    return $true
}

# Function to run KSE experiment
function Run-KSE {
    Write-Host "Running KSE experiment..." -ForegroundColor Green
    Push-Location kse
    try {
        julia --project=../.. main.jl
        Write-Host "KSE experiment completed." -ForegroundColor Green
    }
    catch {
        Write-Host "Error running KSE experiment: $_" -ForegroundColor Red
        return $false
    }
    finally {
        Pop-Location
    }
    return $true
}

# Function to display usage
function Show-Usage {
    Write-Host ""
    Write-Host "Usage: .\run_experiments.ps1 -Experiment <burgers|kse|both>" -ForegroundColor Yellow
    Write-Host "  burgers - Run only the Burgers experiment" -ForegroundColor Cyan
    Write-Host "  kse     - Run only the KSE experiment" -ForegroundColor Cyan
    Write-Host "  both    - Run both experiments" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_experiments.ps1 -Experiment burgers" -ForegroundColor Gray
    Write-Host "  .\run_experiments.ps1 -Experiment kse" -ForegroundColor Gray
    Write-Host "  .\run_experiments.ps1 -Experiment both" -ForegroundColor Gray
}

# Main script logic
$success = $true

switch ($Experiment.ToLower()) {
    "burgers" {
        $success = Run-Burgers
    }
    "kse" {
        $success = Run-KSE
    }
    "both" {
        Write-Host "Running both experiments..." -ForegroundColor Magenta
        $burgersSuccess = Run-Burgers
        Write-Host ""
        $kseSuccess = Run-KSE
        $success = $burgersSuccess -and $kseSuccess
        
        if ($success) {
            Write-Host "All experiments completed successfully!" -ForegroundColor Green
        } else {
            Write-Host "Some experiments failed. Check the output above." -ForegroundColor Red
        }
    }
    default {
        Write-Host "Error: Unknown experiment '$Experiment'" -ForegroundColor Red
        Show-Usage
        exit 1
    }
}

if ($success) {
    Write-Host "Experiment(s) finished successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Experiment(s) failed!" -ForegroundColor Red
    exit 1
}
