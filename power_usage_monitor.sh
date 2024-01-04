#!/bin/bash

# Default values
default_duration=3000  # in seconds
default_output_file="gpu_power_usage.txt"

# Initialize variables with default values
duration=$default_duration
output_file=$default_output_file

# Process command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --time) duration="$2"; shift ;;
        --output) output_file="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Determine the number of GPUs
num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Generate the header
header=$(seq -s ", " 0 $((num_gpus - 1)) | sed 's/[0-9]\+/device-&/g')
echo "$header" >> "$output_file"


# Run watch command in background
watch -n 1 nvidia-smi &

# Save the PID of the watch command
watch_pid=$!


# Calculate the number of loops (assuming a 60-second interval)
loops=$((duration))

# Loop to record data
for ((i=1; i<=loops; i++))
do

    # Write only the power usage number to the file
    # Get the power usage for all GPUs and transform it into a single line
    power_usage=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$/\n/')

    # Write the transformed power usage to the file
    echo "$power_usage" >> "$output_file"
    
    sleep 1  # Wait for 1 minute
done

# Stop the watch command
kill $watch_pid


# echo "Recording ended at $(date)" >> "$output_file"
