import torch
import datasets
import os
import transformers
from BERT_Trainer.Trainer import Trainer




def main():
    # Create the model trainer
    trainer = Trainer(device=torch.device("cuda:0"))
    
    # Train model
    trainer()





if __name__ == "__main__":
    main()