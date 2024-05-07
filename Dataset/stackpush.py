# Internals
from Stack.stack import Stack

# Skip_Langs: Provide List For Specific Languages To Skip, Provide String To Skip All Languages Alphabetically Before String
# For Second Option, Simply Check Which Folder Is Currently Cached And Write The Folder Name

def main():
    cache_dir = "./Stack/models"
    temp_dir_base = ""
    skip_langs = []
    tokenizer = "codellama/CodeLlama-7b-hf"
    truncation = False
    max_length = 1024
    batch_size = 1024
    num_proc = 16
    push_loc = ""
    token = ""

    # Create Callable
    Stack(cache_dir = cache_dir, temp_dir_base = temp_dir_base, 
          skip_langs = skip_langs, tokenizer = tokenizer, 
          truncation = truncation, max_length = max_length, 
          batch_size = batch_size, num_proc = num_proc, 
          push_loc = push_loc, token = token)

if __name__ == "__main__":
    main()
