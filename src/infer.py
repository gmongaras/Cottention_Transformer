import torch
import random
from Transformer import Transformer
from datasets import load_dataset, load_from_disk






def main():
    # Model params
    dim = 512
    num_layers = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    # Training params
    batch_size = 128
    learning_rate = 1e-4
    epochs = 1000
    max_length = 200
    
    
    
    
    # # Load in the text datasets
    # with open(english_path, "r", encoding="utf-8") as f:
    #     english = f.readlines()
    # with open(spanish_path, "r", encoding="utf-8") as f:
    #     spanish = f.readlines()
    
    # # Combine lists
    # dataset = list(zip(english, spanish))
    
    # # Cut dataset to be an even multiple of batch size
    # dataset = dataset[:len(dataset) - (len(dataset) % batch_size)]
    
    
    
    
    # Create the model
    model = Transformer(num_layers, dim).to(device)
    # Load in checkpoint for model
    model.load_state_dict(torch.load("./checkpoints/checkpoint-0-2000.pt"), strict=False)
    
    # '[CLS] After the martyrdom of St. Boniface, Vergilius was made Bishop of Salzburg ( 766 or 767 ) and laboured successfully for the upbuilding of his diocese as well as for the spread of the Faith in neighbouring heathen countries, especially in Carinthia. He died at Salzburg, 27 November, 789. In 1233 he was canonized by Gregory IX. His doctrine that the earth is a sphere was derived from the teaching of ancient geographers, and his belief in the existence of the antipodes was probably influenced by the accounts which the ancient Irish voyagers gave of their journeys. This, at least, is the opinion of Rettberg ( " Kirchengesch. Deutschlands ", II, 236 ). [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'
    
    # Generate a sentence
    # tokenized = model.tokenizer("I", return_tensors="pt").to(device)
    tokenized = {"input_ids": torch.tensor([[101]], dtype=torch.long).to(device)}
    for i in range(0, 128):
        # Tokenize the current sentence
        # tokenized = model.tokenizer(cur, return_tensors="pt").to(device)
        
        # Get the output
        output = model(tokenized)
        output = output[0, -1]
        
        # Don't predict [SEP]
        # output[102] = -1e9
        
        # Sample
        output = torch.softmax(output, dim=-1)
        output_ = torch.multinomial(output, 1)
        
        # Get the next word
        next_word = model.tokenizer.decode(output_)
        
        # Add next word to the input ids
        tokenized["input_ids"] = torch.cat((tokenized["input_ids"], output_.unsqueeze(-1)), dim=-1)
    
    print(model.tokenizer.decode(tokenized["input_ids"][0]))
    
    
    
    
    
    
if __name__ == "__main__":
    main()