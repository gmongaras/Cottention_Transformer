import torch
import datasets
import os
import transformers
from BERT_Trainer.Trainer import Trainer




def main():
    # Create the model trainer
    trainer = Trainer()
    
    # Train model
    trainer()





if __name__ == "__main__":
    main()