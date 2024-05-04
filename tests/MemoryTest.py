import time
import numpy as np
import torch
import transformers
from BERT_Trainer.BertCosAttention import BertCosAttention
from transformers.models.bert.modeling_bert import BertSelfAttention as BertSoftAttention

from GPT_Trainer.GPTCosAttention import GPTCosAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention as GPTSoftAttention


# Make matplotlib more professional
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

# Increase DPI of plots
plt.rcParams['figure.dpi'] = 400


# Different dimension and sequence lengths
dimensions = [1024, 2048, 4096, 8192, 16384]
heads = [1, 2, 4, 8, 16, 32, 64]
lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]



BERT_Config = transformers.BertConfig.from_dict({
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 28996
    })


GPT_Config = transformers.GPTJConfig.from_dict({
        "activation_function": "gelu_new",
        "architectures": [
            "GPTJForCausalLM"
        ],
        "attn_pdrop": 0.0,
        "bos_token_id": 50256,
        "embd_pdrop": 0.0,
        "eos_token_id": 50256,
        "gradient_checkpointing": False,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gptj",
        "n_embd": 1024,
        "n_head": 16,
        "n_inner": 4*768,
        "n_layer": 20,
        "n_positions": 1024,
        "resid_pdrop": 0.0,
        "rotary": True,
        "rotary_dim": 64,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {
            "do_sample": True,
            "max_length": 50,
            "temperature": 1.0
            }
        },
        "tie_word_embeddings": False,
        "tokenizer_class": "GPT2Tokenizer",
        "transformers_version": "4.18.0.dev0",
        "use_cache": True,
        "vocab_size": 50400
    })



def test_single_mem(dim, head, length, mask,  position_ids, model):
    model.eval()
    model.cuda()
    x = torch.randn(1, length, dim).cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=None)
    torch.cuda.reset_accumulated_memory_stats(device=None)
    torch.cuda.reset_max_memory_allocated(device=None)
    torch.cuda.reset_max_memory_cached(device=None)
    mem_bef = torch.cuda.max_memory_allocated()
    try:
        y = model(x, attention_mask=mask, position_ids=position_ids)
    except TypeError:
        y = model(x, attention_mask=mask)
    mem_aft = torch.cuda.max_memory_allocated()
    del x
    del y
    torch.cuda.reset_peak_memory_stats(device=None)
    torch.cuda.empty_cache()
    
    return (mem_aft - mem_bef)/1e9


def test_single_time_(dim, head, length, mask,  position_ids, model):
    # Time the function
    model.eval()
    model.cuda()
    x = torch.randn(1, length, dim).cuda()
    try:
        start = time.time()
        y = model(x, attention_mask=mask, position_ids=position_ids)
        end = time.time()
    except TypeError:
        start = time.time()
        y = model(x, attention_mask=mask)
        end = time.time()
        
    del x
    del y
    torch.cuda.reset_peak_memory_stats(device=None)
    torch.cuda.empty_cache()
        
    return end - start

def test_single_time(dim, head, length, mask,  position_ids, model, num_times=10):
    times = []
    for _ in range(num_times):
        times.append(test_single_time_(dim, head, length, mask,  position_ids, model))
    return sum(times)/len(times)




def test_single_(dim, head, length):
    # Mask is all ones for this sequence length
    attn_mask = torch.ones(1, length).cuda()
    position_ids = torch.arange(length).unsqueeze(0).cuda()
    
    print(f"Dimension: {dim}, Heads: {head}, Length: {length}")
    BERT_Config.num_attention_heads = head
    BERT_Config.hidden_size = dim
    GPT_Config.n_head = head
    GPT_Config.n_embd = dim
    GPT_Config.n_positions = length
    GPT_Config.rotary_dim = min(64, dim//head)
    BERT = BertCosAttention(BERT_Config)
    GPT = GPTCosAttention(GPT_Config)
    BERT_soft = BertSoftAttention(BERT_Config)
    GPT_soft = GPTSoftAttention(GPT_Config)
    BERT_mem = test_single_mem(dim, head, length, attn_mask, position_ids, BERT)
    GPT_mem = test_single_mem(dim, head, length, attn_mask,  position_ids, GPT)
    BERT_soft_mem = test_single_mem(dim, head, length, attn_mask,  position_ids, BERT_soft)
    GPT_soft_mem = test_single_mem(dim, head, length, attn_mask,  position_ids, GPT_soft)
    print(f"BERT: {BERT_mem}, GPT: {GPT_mem}, BERT_soft: {BERT_soft_mem}, GPT_soft: {GPT_soft_mem}")
    BERT_time = test_single_time(dim, head, length, attn_mask, position_ids, BERT)
    GPT_time = test_single_time(dim, head, length, attn_mask,  position_ids, GPT)
    BERT_soft_time = test_single_time(dim, head, length, attn_mask,  position_ids, BERT_soft)
    GPT_soft_time = test_single_time(dim, head, length, attn_mask,  position_ids, GPT_soft)
    del BERT
    del GPT
    del BERT_soft
    del GPT_soft
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=None)
    print()
    
    return BERT_mem, GPT_mem, BERT_soft_mem, GPT_soft_mem, BERT_time, GPT_time, BERT_soft_time, GPT_soft_time



# Iterate over all possible dimension sizes. Keep the number of heads and length constant
head = 16
length = 1024
BERT_mems = []
GPT_mems = []
BERT_soft_mems = []
GPT_soft_mems = []
BERT_times = []
GPT_times = []
BERT_soft_times = []
GPT_soft_times = []
for dim in dimensions:
    BERT_mem, GPT_mem, BERT_soft_mem, GPT_soft_mem, BERT_time, GPT_time, BERT_soft_time, GPT_soft_time = test_single_(dim, head, length)
    BERT_mems.append(BERT_mem)
    GPT_mems.append(GPT_mem)
    BERT_soft_mems.append(BERT_soft_mem)
    GPT_soft_mems.append(GPT_soft_mem)
    BERT_times.append(BERT_time)
    GPT_times.append(GPT_time)
    BERT_soft_times.append(BERT_soft_time)
    GPT_soft_times.append(GPT_soft_time)
    
# Plot the results
plt.plot(dimensions, BERT_mems, label="Cos BERT", c="blue", linestyle="--")
plt.plot(dimensions, GPT_mems, label="Cos GPT", c="blue")
plt.plot(dimensions, BERT_soft_mems, label="Soft BERT", c="red", linestyle="--")
plt.plot(dimensions, GPT_soft_mems, label="Soft GPT", c="red")
plt.xlabel("Dimension")
plt.ylabel("Memory Usage (GB)")
plt.legend()
plt.show()
plt.savefig("tests/dim.png")
plt.clf()

# Plot times
plt.plot(dimensions, BERT_times, label="Cos BERT", c="blue", linestyle="--")
plt.plot(dimensions, GPT_times, label="Cos GPT", c="blue")
plt.plot(dimensions, BERT_soft_times, label="Soft BERT", c="red", linestyle="--")
plt.plot(dimensions, GPT_soft_times, label="Soft GPT", c="red")
plt.xlabel("Dimension")
plt.ylabel("Time (s)")
plt.legend()
plt.show()
plt.savefig("tests/dim_time.png")
plt.clf()



# Iterate over all possible sequence lengths. Keep the number of heads and dimension constant
head = 16
dim = 2048
BERT_mems = []
GPT_mems = []
BERT_soft_mems = []
GPT_soft_mems = []
GPT_flash_mems = []
BERT_times = []
GPT_times = []
BERT_soft_times = []
GPT_soft_times = []
for length in lengths:
    BERT_mem, GPT_mem, BERT_soft_mem, GPT_soft_mem, BERT_time, GPT_time, BERT_soft_time, GPT_soft_time = test_single_(dim, head, length)
    BERT_mems.append(BERT_mem)
    GPT_mems.append(GPT_mem)
    BERT_soft_mems.append(BERT_soft_mem)
    GPT_soft_mems.append(GPT_soft_mem)
    BERT_times.append(BERT_time)
    GPT_times.append(GPT_time)
    BERT_soft_times.append(BERT_soft_time)
    GPT_soft_times.append(GPT_soft_time)

# Plot the results
plt.plot(lengths, BERT_mems, label="Cos BERT", c="blue", linestyle="--")
plt.plot(lengths, GPT_mems, label="Cos GPT", c="blue")
plt.plot(lengths, BERT_soft_mems, label="Soft BERT", c="red", linestyle="--")
plt.plot(lengths, GPT_soft_mems, label="Soft GPT", c="red")
plt.xlabel("Length")
plt.ylabel("Memory Usage (GB)")
plt.legend()
plt.show()
plt.savefig("tests/length.png")
plt.clf()

# Plot times
plt.plot(lengths, np.log(BERT_times), label="Cos BERT", c="blue", linestyle="--")
plt.plot(lengths, np.log(GPT_times), label="Cos GPT", c="blue")
plt.plot(lengths, np.log(BERT_soft_times), label="Soft BERT", c="red", linestyle="--")
plt.plot(lengths, np.log(GPT_soft_times), label="Soft GPT", c="red")
plt.xlabel("Length")
plt.ylabel("Log Time (s)")
plt.legend()
plt.show()
plt.savefig("tests/length_time.png")
plt.clf()






# Iterate over all possible head sizes. Keep the sequence length and dimension constant
dim = 4096
length = 1024
BERT_mems = []
GPT_mems = []
BERT_soft_mems = []
GPT_soft_mems = []
BERT_times = []
GPT_times = []
BERT_soft_times = []
GPT_soft_times = []
for head in heads:
    BERT_mem, GPT_mem, BERT_soft_mem, GPT_soft_mem, BERT_time, GPT_time, BERT_soft_time, GPT_soft_time = test_single_(dim, head, length)
    BERT_mems.append(BERT_mem)
    GPT_mems.append(GPT_mem)
    BERT_soft_mems.append(BERT_soft_mem)
    GPT_soft_mems.append(GPT_soft_mem)
    BERT_times.append(BERT_time)
    GPT_times.append(GPT_time)
    BERT_soft_times.append(BERT_soft_time)
    GPT_soft_times.append(GPT_soft_time)
    
# Plot the results
plt.plot(heads, BERT_mems, label="Cos BERT", c="blue", linestyle="--")
plt.plot(heads, GPT_mems, label="Cos GPT", c="blue")
plt.plot(heads, BERT_soft_mems, label="Soft BERT", c="red", linestyle="--")
plt.plot(heads, GPT_soft_mems, label="Soft GPT", c="red")
plt.xlabel("Heads")
plt.ylabel("Memory Usage (GB)")
plt.legend()
plt.show()
plt.savefig("tests/head.png")
plt.clf()

# Plot times
plt.plot(heads, BERT_times, label="BERT", c="blue", linestyle="--")
plt.plot(heads, GPT_times, label="Cos GPT", c="blue")
plt.plot(heads, BERT_soft_times, label="Soft BERT", c="red", linestyle="--")
plt.plot(heads, GPT_soft_times, label="Soft GPT", c="red")
plt.xlabel("Heads")
plt.ylabel("Time (s)")
plt.legend()
plt.show()
plt.savefig("tests/head_time.png")
plt.clf()