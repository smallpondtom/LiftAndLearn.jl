# Streaming-OpInf Experiments

This directory contains scripts to run Streaming-OpInf experiments on the 1D Burgers' equation and the Kuramoto-Sivashinsky equation (KSE). The experiments demonstrate the performance of the Streaming-OpInf method for learning reduced-order models from streaming data. The 3D channel data is not available for public release and thus experiments on that dataset are not included here.

## Running the Experiments

To run the experiments, first clone this repository in the **stream** branch

```bash
git clone -b stream https://github.com/smallpondtom/LiftAndLearn.jl.git
```

Then navigate to the `scripts/Streaming-OpInf` directory with the command

```bash
cd LiftAndLearn.jl/scripts/Streaming-OpInf
```

and execute the appropriate script for your operating system:

- For Windows PowerShell:
```powershell
.\run_experiments.ps1 -Experiment burgers
.\run_experiments.ps1 -Experiment kse
.\run_experiments.ps1 -Experiment both
```

- For Windows Command Prompt:
```cmd
run_experiments.bat burgers
run_experiments.bat kse
run_experiments.bat both
```

- For Unix-based systems (Linux, macOS):
```bash
chmod +x ./run_experiments.sh burgers
chmod +x ./run_experiments.sh kse
chmod +x ./run_experiments.sh both
```

It is going to take a while to run each experiment so maybe run them overnight.