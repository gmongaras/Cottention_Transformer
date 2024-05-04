# Internals
from Pile.pile import Pile

# For Paper Results, Use "EleutherAI/gpt-j-6B" Tokenizer
# Job Index -1 For Single Core Processing, Saving Each Subset Individually (Linearly)
# Total Jobs (n > 0) For Multi Core Processing (Set Job Index Accordingly, Incrementally)
# Can Use Process Scheduler For Multi Core Processing (SLURM Script)

# Process = True: Process, Save Each Subset Individually
# Push = True: Combine All Subsets Into Single Dataset, Push To Hugging Face Hub
# If Parallelized, Make Sure To Keep Push False And Then Submit Another Job That Combines All Subsets
# In Other Words, Push Cannot Be Parallelized - But Process Can Be`
# If Individual (Job Index -1), Process And Push In One Go

def main():
    process = True
    push = True
    cache_dir = "./Pile/models"
    data_dir = ""
    tokenizer = "meta-llama/Llama-2-7b-hf"
    job_index = -1
    total_jobs = 64
    truncation = False
    max_length = 1024
    batch_size = 1024
    num_proc = 16
    num_shards = 64
    push_loc = ""
    token = ""

    # Create Callable
    Pile(process = process, push = push, cache_dir = cache_dir, data_dir = data_dir, tokenizer = tokenizer,
         job_index = job_index, total_jobs = total_jobs, truncation = truncation, max_length = max_length,
         batch_size = batch_size, num_proc = num_proc, num_shards = num_shards, push_loc = push_loc, token = token)()

if __name__ == "__main__":
    main()
