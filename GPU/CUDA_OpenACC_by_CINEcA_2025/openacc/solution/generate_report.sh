#!/bin/bash

# Define the directory to store reports
report_dir_nsys="reports/nsys"
report_dir_ncu="reports/ncu"

# Create the directory if it doesn't exist
mkdir -p "$report_dir_nsys"
mkdir -p "$report_dir_ncu"

# Generate a unique report file name based on the current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
report_file="report_${timestamp}"

# Run the nsys profile command and store the report in the specified directory
nsys profile --trace=cuda,nvtx,openacc --stats=true -o "${report_dir_nsys}/${report_file}" "$@"

# Notify the user
echo "Report generated: ${report_dir_nsys}/${report_file}.qdrep"

# Run the ncu profile command and store the report in the specified directory
ncu --set full --launch-skip 5 --launch-count 1 --import-source yes -o "${report_dir_ncu}/${report_file}" "$@"

