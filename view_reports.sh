#!/bin/bash
# Wrapper script to view reports from any location

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to project root
cd "$SCRIPT_DIR"

# Run the view reports script
bash .kiro/scripts/view_reports.sh "$@"
