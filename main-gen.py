import subprocess
import time
import os
import pandas as pd

# Parameters
output_dir = "output"
width = 1080
height = 1080
batch = 1
prompt_file = 'prompt-bank.xlsx'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load prompts from Excel file
prompts_df = pd.read_excel(prompt_file)

# GPU configurations
gpu_scripts = [
    ("gpu0.py", "gpu0_output.png"),
    ("gpu1.py", "gpu1_output.png"),
    ("gpu2.py", "gpu2_output.png")
]

# Start timing
start_time = time.time()

# Process each prompt
for index, row in prompts_df.iterrows():
    pos = row['Positive Prompt']
    neg = row['Negative Prompt']

    # Launch processes for all GPUs for the current prompt
    processes = []
    for gpu_id, (script, gpu_output_file) in enumerate(gpu_scripts):
        output_file = f"output_{index}_gpu{gpu_id}.png"
        output_path = os.path.join(output_dir, output_file)
        cmd = [
            '/bin/python3', script,
            '--pos', pos,
            '--neg', neg,
            '--output', output_path,
            '--width', str(width),
            '--height', str(height),
            '--batch', str(batch)
        ]
        process = subprocess.Popen(cmd)
        processes.append(process)

    # Wait for all processes to complete before moving to the next row
    for process in processes:
        process.wait()

# Calculate elapsed time
elapsed_time = time.time() - start_time

print("All processes completed.")
print(f"Total time taken: {elapsed_time:.2f} seconds")

# Example Test Command
# /bin/python3 parallel_gpu_runner.py
