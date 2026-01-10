#!/bin/bash 

# Usage
# Change directory to scripts/Streaming-OpInf then run:
# chmod +x ./run_experiments.sh burgers
# chmod +x ./run_experiments.sh kse
# chmod +x ./run_experiments.sh both

# Function to run Burgers experiment
run_burgers() {
    echo "Running Burgers experiment..."
    cd burgers
    julia --project=../.. main.jl
    cd ..
    echo "Burgers experiment completed."
}

# Function to run KSE experiment
run_kse() {
    echo "Running KSE experiment..."
    cd kse
    julia --project=../.. main.jl
    cd ..
    echo "KSE experiment completed."
}

# Function to display usage
usage() {
    echo "Usage: $0 [burgers|kse|both]"
    echo "  burgers - Run only the Burgers experiment"
    echo "  kse     - Run only the KSE experiment"
    echo "  both    - Run both experiments"
    exit 1
}

# Main script logic
if [ $# -eq 0 ]; then
    echo "Error: No experiment specified."
    usage
fi

case "$1" in
    "burgers")
        run_burgers
        ;;
    "kse")
        run_kse
        ;;
    "both")
        echo "Running both experiments..."
        run_burgers
        echo ""
        run_kse
        echo "All experiments completed."
        ;;
    *)
        echo "Error: Unknown experiment '$1'"
        usage
        ;;
esac

echo "Experiment(s) finished successfully!"