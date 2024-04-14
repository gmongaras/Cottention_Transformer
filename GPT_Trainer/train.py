import torch
import datasets
import os
import transformers
try:
    from GPT_Trainer.Trainer import Trainer
except ModuleNotFoundError:
    from Trainer import Trainer




def main():
    # Create the model trainer
    batch_size=128
    learning_rate=1e-4
    warmup_steps=10_000
    num_steps=1_000_000
    dev="gpu"
    wandb_name="del"
    log_steps=10
    use_amp=True
    attention_type="cos" # cos, or soft
    clipping_value=None
    weight_decay=0.01
    model_save_path = "models_GPT/del"
    # model_save_path = "models/del"
    num_save_steps = 10_000
    keep_dataset_in_mem = False
    model_max_length = 1024
    
    # Load in a checkpoint
    load_checkpoint = False
    checkpoint_path = "models_GPT/del/"
    
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
        model_save_path=model_save_path,
        num_save_steps=num_save_steps,
        keep_dataset_in_mem=keep_dataset_in_mem,
        load_checkpoint=load_checkpoint,
        checkpoint_path=checkpoint_path,
        model_max_length=model_max_length
    )
    
    # Train model
    trainer()





if __name__ == "__main__":
    main()
