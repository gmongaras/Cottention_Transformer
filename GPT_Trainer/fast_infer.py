import torch
from torch import nn
import transformers
import datasets
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext


import numpy as np


try:
    from GPT_Trainer.GPTCosAttention import GPTCosAttention
except ModuleNotFoundError:
    from GPTCosAttention import GPTCosAttention






@torch.no_grad()
def infer():
    # Path to the model
    # model_path = "models/SM AdamW"
    # attention_type = "soft"
    # model_path = "models/redo_lr1e-4_SM"
    model_path = "models_GPT/Cosine"
    attention_type = "cos"
    fast_inference= True
    get_mem = True
    device = "cpu"
    
    
    # Load the model
    model = transformers.GPTJForCausalLM.from_pretrained(model_path.replace(" ", "_"))
    model.eval()
    
    
    # Replace all self attention layers (BertSelfAttention) with the cosine attention layer (GPTCosAttention)
    if attention_type == "cos":
        model.config.fast_inference = fast_inference
        
        for layer in model.transformer.h:
            old = layer.attn
            layer.attn = GPTCosAttention(model.config).to(layer.attn.q_proj.weight.device)
            
            # Copy weights
            layer.attn.q_proj.weight.data = old.q_proj.weight.data
            if old.q_proj.bias is not None:
                layer.attn.q_proj.bias.data = old.q_proj.bias.data
            else:
                layer.attn.q_proj.bias = None
            layer.attn.k_proj.weight.data = old.k_proj.weight.data
            if old.k_proj.bias is not None:
                layer.attn.k_proj.bias.data = old.k_proj.bias.data
            else:
                layer.attn.k_proj.bias = None
            layer.attn.v_proj.weight.data = old.v_proj.weight.data
            if old.v_proj.bias is not None:
                layer.attn.v_proj.bias.data = old.v_proj.bias.data
            else:
                layer.attn.v_proj.bias = None
            layer.attn.out_proj.weight.data = old.out_proj.weight.data
            if old.out_proj.bias is not None:
                layer.attn.out_proj.bias.data = old.out_proj.bias.data
            else:
                layer.attn.out_proj.bias = None
            
            del old
            
        # Load extra params if needed
        model.load_state_dict(torch.load(model_path.replace(" ", "_") + "/pytorch_model.bin", map_location=model.transformer.h[0].attn.q_proj.weight.device), strict=False)
        
        # Clear cache
        torch.cuda.empty_cache()
        
    # Number of parameters in billions
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000_000
    print(f"Number of parameters: {num_params:.2f}B")
        
    model = model.to(device)
        
    # Load the tokenizer
    tokenizer = torch.load(os.path.join(model_path, "tokenizer.pt"))
            
    # inference
    sentence = "This is some text that"
    # sentence = r"""He noted that the style was both a "physical workout", the core muscles constantly working to keep the body balanced on the board, and "an exercise in mental focus"[SEP]When he lost focus as he had often done on his yoga mat, his board "penaliz[ed him] for letting [his] mind wander" and, like what the instructor had described as "only about 10% of her students", he fell into the "chilly" water"""
    # sentence = """
    # Cosine attention or cottention is
    # """.strip()
    
    # sentence = """
    # "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language[SEP]Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful"
    # """.strip()
    
    
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if get_mem:
        mems = []
    torch.cuda.empty_cache()
    
    # Initial seed
    # Feed the sentences through the model to get the next word
    probs = model(**inputs).logits[0, -1]
    dist = torch.distributions.Categorical(logits=probs)
    next_word = dist.sample()
    word = tokenizer.decode(next_word.unsqueeze(0))
    print(word, end="")
    
    
    s = len(inputs["input_ids"][0])
    for i in range(s, 1024):
        # New inputs is just the next word
        inputs = tokenizer(word, return_tensors="pt")
        inputs["position_ids"] = torch.tensor([[i]]).to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get the logits
        probs = model(**inputs).logits[0, -1]
        
        # Set prob of <|endoftext|> to 0
        # probs[50256] = -float("inf")
        
        # Get the predicted next word
        dist = torch.distributions.Categorical(logits=probs)
        next_word = dist.sample()
        
        if next_word == 50256:
            break
        
        word = tokenizer.decode(next_word.unsqueeze(0))
        print(word, end="")
        
        
        # Collect memory usage
        if get_mem:
            torch.cuda.empty_cache()
            mems.append(torch.cuda.memory_allocated())
        
    # How much memory is being used
    print(torch.cuda.memory_summary())
    
    # Save memory usage as numpy array
    if get_mem:
        np.save("mem.npy", np.array(mems))
    
    
    
if __name__ == "__main__":
    infer()