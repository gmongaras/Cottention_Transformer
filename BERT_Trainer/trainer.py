import torch
import datasets
import os
import transformers
try:
    from BERT_Trainer.Trainer import Trainer
except ModuleNotFoundError:
    from Trainer import Trainer




def main():
    # Create the model trainer
    trainer = Trainer(dev="gpu")
    
    # Train model
    trainer()





if __name__ == "__main__":
    main()