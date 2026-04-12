#!/bin/bash

###############################################################################
# movementModel.sh
#
# Master script for the movement model generation. We first load the fit parameters
# and save them to a csv file and then run simulations and make diagnostic plots
# with these parameters. The parameters were fit using a combination of quantitative 
# parameter searches and manual adjustments, which is why we supply the fit parameters. 
# However, to see the code for the quantitative parameter sweeps, see 01_fitParameters.py. 

# Steps:
#   1. fit parameters
#   2. run simulations and plot diagnostics
#
# Usage:
#   ./movementModel.sh --config path/to/config.py
#
# Arguments:
#   --config : Path to the Python configuration file. This file must define 
#              a variable "home_directory" that points to the root directory.
###############################################################################

set -e

# parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config)
      CONFIG_FILE="$2"
      shift 
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# check if config file was provided
if [[ -z "$CONFIG_FILE" ]]; then
  echo "Error: --config path/to/config.py is required"
  exit 1
fi

# get the home directory from config.py using Python
home_dir=$(python3 - <<END
import re
with open("$CONFIG_FILE") as f:
    txt = f.read()
match = re.search(r'home_directory\s*=\s*"(.*)"', txt)
if match:
    print(match.group(1))
END
)

# check if we got it
if [ -z "$home_dir" ]; then
  echo "Error: Could not parse home_directory from $CONFIG_FILE"
  exit 1
fi

echo "Setting home directory to: $home_dir"

# running the scriptst
echo "-------------------------------------------------------------------------------------"
echo "Running 01_fitParameters.py"
python "$(dirname "$CONFIG_FILE")/code/03_movementModel/01_fittingParameters.py" "$home_dir"
echo "Completed 01_fitParameters.py"
echo "-------------------------------------------------------------------------------------"
echo "Running 02_simulationDiagnostics.py"
python "$(dirname "$CONFIG_FILE")/code/03_movementModel/02_simulationDiagnostics.py" "$home_dir"
echo "Completed 02_simulationDiagnostics.py"
echo "-------------------------------------------------------------------------------------"

