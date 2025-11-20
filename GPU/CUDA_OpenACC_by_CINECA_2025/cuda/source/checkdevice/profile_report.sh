#!/bin/bash

# Define the directory to store reports
report_dir="reports"

# Create the directory if it doesn't exist
mkdir -p "$report_dir"

# Generate a unique report file name based on the current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
report_file_base="${report_dir}/report_${timestamp}"

# Check if nsys or ncu should be used (pass the tool name as the first argument)
if [[ "$1" == "nsys" ]]; then
    shift  # Remove the first argument (nsys) and pass the rest to the command
    # Run the nsys profile command and store the report
    nsys profile --trace=cuda,nvtx --stats=true -o "${report_file_base}_nsys" "$@"
    echo "NSYS report generated: ${report_file_base}_nsys.qdrep"
elif [[ "$1" == "ncu" ]]; then
    shift  # Remove the first argument (ncu) and pass the rest to the command
    # Run the ncu command with the specified options
    ncu -o "${report_file_base}_ncu" "$@"
    echo "NCU report generated: ${report_file_base}_ncu.ncu-rep"
elif [[ "$1" == "rooftop" ]]; then
    shift  # Remove the first argument (ncu) and pass the rest to the command
    # Run the ncu command with the specified options
    ncu --set roofline --section SpeedOfLight -o "${report_file_base}_ncu" "$@"
    echo "NCU report generated: ${report_file_base}_ncu.ncu-rep"
else
    # If no argument is provided, run both nsys and ncu
    echo "No profiling tool specified, running both nsys and ncu."

    # Run nsys profile command and store the report
    nsys profile --trace=cuda,nvtx --stats=true -o "${report_file_base}_nsys" "$@"
    echo "NSYS report generated: ${report_file_base}_nsys.qdrep"

    # Run ncu command with the specified options and save in the reports folder
    ncu -o "${report_file_base}_ncu" "$@"
    echo "NCU report generated: ${report_file_base}_ncu.ncu-rep"
 fi
