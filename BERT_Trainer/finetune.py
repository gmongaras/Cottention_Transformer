import torch
import datasets
import os
import transformers
try:
    from BERT_Trainer.Trainer import Trainer
except ModuleNotFoundError:
    from Trainer import Trainer




import click

@click.command()
@click.option('--batch_size', default=32, help='Batch size for the model.')
@click.option('--learning_rate', default=5e-5, help='Learning rate for the model.')
@click.option('--warmup_steps', default=10000, help='Number of warmup steps.')
@click.option('--num_steps', default=1000000, help='Total number of training steps.')
@click.option('--dev', default='gpu', help='Device to use for training (e.g., gpu or cpu).')
@click.option('--wandb_name', default=None, help='Weights & Biases project name.')
@click.option('--log_steps', default=10, help='Frequency of logging steps.')
@click.option('--use_amp', is_flag=True, help='Use automatic mixed precision.')
@click.option('--attention_type', default='soft', help='Type of attention to use.')
@click.option('--clipping_value', default=None, type=float, help='Gradient clipping value.')
@click.option('--weight_decay', default=0.01, help='Weight decay for the optimizer.')
@click.option('--model_save_path', default='models/SM_Finetune', help='Path to save the trained model.')
@click.option('--num_save_steps', default=10000, help='Frequency of saving model checkpoints.')
@click.option('--keep_dataset_in_mem', is_flag=True, help='Keep the entire dataset in memory.')
@click.option('--finetune_task', help='Task for finetuning the model.', required=True)
@click.option('--checkpoint_path', help='Path to the model checkpoint for loading.', required=True)

def main(batch_size, learning_rate, warmup_steps, num_steps, dev, wandb_name, log_steps, use_amp, attention_type, clipping_value, weight_decay, model_save_path, num_save_steps, keep_dataset_in_mem, finetune_task, checkpoint_path):
    if wandb_name is not None:
        wandb_name=f"{finetune_task}_{wandb_name}_{learning_rate}_{batch_size}"
        
    # # Create the model trainer
    # batch_size=32
    # learning_rate=5e-5
    # warmup_steps=10_000
    # num_steps=1_000_000
    # dev="gpu"
    # wandb_name=f"QQP_SM_Finetune"
    # log_steps=10
    # use_amp=True
    # attention_type="soft"
    # clipping_value=None
    # weight_decay=0.01
    # model_save_path = "models/SM_Finetune"
    # # model_save_path = "models/del"
    # num_save_steps = 10_000
    # keep_dataset_in_mem = False
    # finetune_task = "qqp"
    
    # Load in a checkpoint
    # checkpoint_path = "models/redo_lr1e-4_SM/"
    # checkpoint_path = "models/redo_lr1e-4_Cos_Div/"
    
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
        load_checkpoint=True,
        checkpoint_path=checkpoint_path,
        finetune=True,
        finetune_task=finetune_task,
    )
    
    # Train model
    trainer.finetune()





if __name__ == "__main__":
    main()
