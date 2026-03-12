#!/bin/bash

###############################################################################
# stateClassification.sh
#
# Master script for assigning states to the lynx trajectory data. We set up a 
# two state model based on the behavior of the lynx. State 1 represents the local 
# diffusion within a home territory, whereas state 2 represents the long excursions
# the lynx will embark on.

# In step 3a + 3b, we address the loops that seem to be somewhat common. Lynx will embark
# on excursions i.e. switch into state 2, but ultimately return back home. Since 
# state 1 will already be simualted with a pull towards X_home, we simulate the looping
# behavior by having some probability of updating X_home when we switch from state 
# 2 to 1 that depends on the distance from X_home. If we update X_home, the lynx
# will now have a pull towards its new home so it successfully switched territory. 
# If we don't update X_home, the pull towards home will pull it back towards its 
# original territory, creating loops. This section will only become relevant in the 
# next steps when we talk about simulating the movement, but that is why step 3
# colors the returning loop as state 3, which we will eventually simulate as state 1.
#
# Steps:
#   1. Calculate WMSD
#   2. Assign States (1 and 2)
#   3. Split loops into outbound and inbound (relevant for later simulation)
#
# Usage:
#   ./stateClassification.sh --config path/to/config.py
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
echo "Running 01_wmsdCalculation.py"
python "$(dirname "$CONFIG_FILE")/code/02_stateClassification/01_wmsdCalculation.py" "$home_dir"
echo "Completed 01_wmsdCalculation.py"
echo "-------------------------------------------------------------------------------------"
echo "Running 02a_stateClassification.py"
python "$(dirname "$CONFIG_FILE")/code/02_stateClassification/02a_stateClassification.py" "$home_dir"
echo "Completed 02a_stateClassification.py"
echo "-------------------------------------------------------------------------------------"
echo "Running 02b_stateClassification_diagnostics.py"
python "$(dirname "$CONFIG_FILE")/code/02_stateClassification/02b_stateClassification_diagnostics.py" "$home_dir"
echo "Completed 02b_stateClassification_diagnostics.py"
echo "-------------------------------------------------------------------------------------"
echo "Running 03a_splittingLoops.py"
python "$(dirname "$CONFIG_FILE")/code/02_stateClassification/03a_splittingLoops.py" "$home_dir"
echo "Completed 03a_splittingLoops.py"
echo "-------------------------------------------------------------------------------------"
echo "Running 03b_loopDiagnostics.py"
python "$(dirname "$CONFIG_FILE")/code/02_stateClassification/03b_loopDiagnostics.py" "$home_dir"
echo "Completed 03b_loopDiagnostics.py"

