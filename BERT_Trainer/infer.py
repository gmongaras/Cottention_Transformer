import torch
from torch import nn
import transformers
import datasets
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext


try:
    from BERT_Trainer.BertCosAttention import BertCosAttention
except ModuleNotFoundError:
    from BertCosAttention import BertCosAttention






def infer():
    # Path to the model
    # model_path = "models/SM AdamW"
    # attention_type = "soft"
    model_path = "models/lr1e-4_Cos_divLearnExpUnif"
    attention_type = "cos"
    
    
    # Load the model
    model = transformers.BertForPreTraining.from_pretrained(model_path.replace(" ", "_"))
    model.eval()
    
    
    # Replace all self attention layers (BertSelfAttention) with the cosine attention layer (BertCosAttention)
    if attention_type == "cos":
        for layer in model.bert.encoder.layer:
            old = layer.attention.self
            layer.attention.self = BertCosAttention(model.config).to(layer.attention.self.query.weight.device)
            
            # Copy weights
            layer.attention.self.query.weight.data = old.query.weight.data
            layer.attention.self.query.bias.data = old.query.bias.data
            layer.attention.self.key.weight.data = old.key.weight.data
            layer.attention.self.key.bias.data = old.key.bias.data
            layer.attention.self.value.weight.data = old.value.weight.data
            layer.attention.self.value.bias.data = old.value.bias.data
            
            del old
            
    # Load extra params if needed
    model.load_state_dict(torch.load(model_path.replace(" ", "_") + "/pytorch_model.bin", map_location=model.bert.encoder.layer[0].attention.self.query.weight.device))
    
    # Load the tokenizer
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_path.replace(" ", "_"))
            
    # inference
    sentence = "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language[SEP]Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful"
    # sentence = r"""He noted that the style was both a "physical workout", the core muscles constantly working to keep the body balanced on the board, and "an exercise in mental focus"[SEP]When he lost focus as he had often done on his yoga mat, his board "penaliz[ed him] for letting [his] mind wander" and, like what the instructor had described as "only about 10% of her students", he fell into the "chilly" water"""
    
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs["token_type_ids"][0][torch.where(inputs["input_ids"][0] == tokenizer.sep_token_id)[0][0]+1:] = 1
    
    # Get the masked token
    # masked_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    masked_index = -23
    inputs["input_ids"][0][masked_index] = torch.tensor(tokenizer.mask_token_id)
    
    encoded = tokenizer.decode(inputs["input_ids"][0])
    
    # Get the logits
    if attention_type == "cos":
        outputs = model(**inputs)
    else:
        outputs = model(**inputs, output_attentions=True)
        
        for attn in outputs.attentions:
            # Matplotlib attention heatmap
            import matplotlib.pyplot as plt
            probs = attn[0].detach().cpu().numpy()
            for head in range(probs.shape[0]):
                # Shape is (num_heads, seq_len, seq_len)
                plt.imshow(probs[head])
                plt.show()
                if not os.path.exists("imgs"):
                    os.makedirs("imgs")
                plt.savefig(f"imgs/attention{head}.png")
                
            print()
    
    # Get the logits for the masked token
    logits = outputs.prediction_logits[0, masked_index]
    NSP_logits = outputs.seq_relationship_logits[0]
    
    # Get the top 5 tokens
    top_5_tokens = torch.topk(logits, 15).indices
    
    # Get the top 5 tokens in string form
    top_5_tokens_str = tokenizer.decode(top_5_tokens)
    
    # Print the results
    print(f"Top 5 tokens: {top_5_tokens_str}")
    
    # Fill in the blank
    encoded2 = encoded.replace("[MASK]", tokenizer.decode(top_5_tokens[:1]))
    
    print(f"Original: {encoded}")
    print(f"Filled in: {encoded2}")
    print(f"NSP prediction: {torch.argmax(NSP_logits)}")
    
    
    
if __name__ == "__main__":
    infer()