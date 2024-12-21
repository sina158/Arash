import argparse
from comfy_script.runtime import *
load('http://127.0.0.1:8189/')

from comfy_script.runtime.nodes import *
import json
import random
import time
import os 

# Argument parsing
parser = argparse.ArgumentParser(description="GPU Image Generation Script")
parser.add_argument('--seed', type=int, default=random.randint(0, 2**32 - 1), help='Random seed for generation')
parser.add_argument('--width', type=int, default=1080, help='Width of the image')
parser.add_argument('--height', type=int, default=1080, help='Height of the image')
parser.add_argument('--batch', type=int, default=1, help='Batch size')
parser.add_argument('--pos', type=str, required=True, help='Positive prompt')
parser.add_argument('--neg', type=str, required=True, help='Negative prompt')
parser.add_argument('--output', type=str, required=True, help='Output file path')

args = parser.parse_args()

# Parameters
seed = args.seed
width = args.width
height = args.height
batch = args.batch
pos = args.pos
neg = args.neg
output_path = args.output

# Start timing
start_time = time.time()
# Load workflow
with open('/home/arash/ComfyUI/user/default/workflows/NetDistAdvancedV2.json', 'r') as f:
    workflow_json = json.load(f)

with Workflow():
    remote_chain = RemoteChainStart(workflow_json, 'on_change', 1, 997380555113050)
    remote_chain, remote_info = RemoteQueueWorker(
        remote_chain, 'http://127.0.0.1:8188/', 1, 'false', 'final_image'
    )
    model, clip, vae = CheckpointLoaderSimple('v1-5-pruned-emaonly.ckpt')
    conditioning = CLIPTextEncode(pos, clip)
    conditioning2 = CLIPTextEncode(neg, clip)
    latent = EmptyLatentImage(width, height, batch)
    latent = KSampler(model, seed, 21, 9, 'euler_ancestral', 'normal', conditioning, conditioning2, latent, 1.0)
    image = VAEDecode(latent, vae)
    SaveImage(image, output_path)

# Wait for process completion by monitoring progress bar
process_complete = False
progress_bar = 0
while not process_complete:
    time.sleep(1)
    # Check for progress updates in terminal output
    if progress_bar >= 21:  # Total steps in progress bar
        process_complete = True
    else:
        progress_bar += 1

# Calculate elapsed time
elapsed_time = time.time() - start_time

# Confirm completion
print(f"Process completed. File saved at: {output_path}")
print(f"Time taken: {elapsed_time:.2f} seconds")
