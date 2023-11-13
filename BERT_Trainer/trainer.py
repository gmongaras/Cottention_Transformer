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
    batch_size=256
    learning_rate=1e-5
    warmup_steps=10_000
    num_steps=1_000_000
    dev="cpu"
    wandb_name="SM learning rate 1e-5"
    log_steps=10
    use_amp=True
    attention_type="cos"
    clipping_value=None
    weight_decay=0.01
    
    trainer = Trainer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_steps=num_steps,
        dev=dev,
        wandb_name=wandb_name,
        log_steps=log_steps,
        use_amp=use_amp,
        attention_type=attention_type,
        clipping_value=clipping_value,
        weight_decay=weight_decay,
    )
    
    # Train model
    trainer()





if __name__ == "__main__":
    main()