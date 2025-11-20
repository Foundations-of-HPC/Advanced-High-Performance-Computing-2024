#!/bin/bash

# Define the directory to store reports
report_dir="reports"

# Create the directory if it doesn't exist
mkdir -p "$report_dir"

# Generate a unique report file name based on the current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
report_file="report_${timestamp}"

# Run the nsys profile command and store the report in the specified directory
nsys profile --trace=cuda,nvtx,openacc --stats=true -o "${report_dir}/${report_file}" "$@"

# Notify the user
echo "Report generated: ${report_dir}/${report_file}.qdrep"

