# Import Statements
import os
import argparse
from datasets import load_dataset, Dataset
from huggingface_hub import login

# Initialize Argument Parser
# Running: python stack.py --token token --dir dir (EX: batches) --batch_size batch_size (EX: 2000000)
parser = argparse.ArgumentParser(description = 'Stream Data From The Stack (HuggingFace Dataset)')
parser.add_argument('--token', type = str, required = True, help = 'HuggingFace Token')
parser.add_argument('--dir', type = str, required = True, help = 'Directory For Saving Batches')
parser.add_argument('--batch_size', type = int, required = True, help = 'Batch Size (Check Memory)')
args = parser.parse_args()

# Login
login(token = args.token)

# Directory Setup
if not os.path.exists(args.dir):
    os.makedirs(args.dir)

# Language Names
file_path = os.path.join(args.dir, "names.txt")

# Process Language Names
with open(file_path, 'r') as file:
    for line in file:
        lang = line.strip()
        if lang:
            
            # Initialize Batch Information
            current_batch = []
            batch_count = 0
            
            # Try Streaming
            try:
                ds = load_dataset("bigcode/the-stack", data_dir = f"data/{lang}", streaming = True, split = "train")
                
                # Append Streamed Samples
                for sample in iter(ds):
                    current_batch.append(sample["content"])
                    
                    # Batch Once Limit Reached
                    if len(current_batch) >= args.batch_size:
                        batch_dataset = Dataset.from_dict({"content": current_batch})
                        save_path = os.path.join(args.dir, f"{lang}_batch_{batch_count}")
                        batch_dataset.save_to_disk(save_path)
                        
                        # Clear Current Batch, Update Batch Counter
                        current_batch = []
                        batch_count += 1

                # Handle Remaining Samples Within Current Batch
                if current_batch:
                    batch_dataset = Dataset.from_dict({"content": current_batch})
                    save_path = os.path.join(args.dir, f"{lang}_batch_{batch_count}")
                    batch_dataset.save_to_disk(save_path)
            
            # Run Exception, Output Error, Continue To Next Language
            except Exception as e:
                print(f"Error Processing Language: {lang}. Reason: {str(e)}")
                continue

# Print Finish
print("Data Streaming Completed!")
