#!/bin/bash

# File to store the power usage data
output_file="gpu_power_usage.txt"

# Record the date and time at the start
echo "Recording started at $(date)" > "$output_file"

# Loop to record data every minute for 60 minutes
for i in {1..60*12}
do
    # Write only the power usage number to the file
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits >> "$output_file"
    
    # Display the full nvidia-smi output on the screen
    nvidia-smi

    sleep 5  # Wait for 1 minute
done

echo "Recording ended at $(date)" >> "$output_file"
