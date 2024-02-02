import torch
import datasets
import os
import transformers
try:
    from BERT_Trainer.Trainer import Trainer
except ModuleNotFoundError:
    from Trainer import Trainer
import time
import gc





def test_model(checkpoint_path, batch_sizes, seq_lens, use_amp=True, train=True):
    model = Trainer(
        use_amp=use_amp,
        dev="gpu",
        keep_dataset_in_mem=False,
        load_checkpoint=True,
        checkpoint_path=checkpoint_path,
    ).model.module
    if train:
        model.train()
    else:
        model.eval()
    config = model.config
    
    
    def test(batch_size, seq_len):
        # Clean memory cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Time the time to do a forward pass
        dummy = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long).to(model.device),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long).bool().to(model.device),
            "token_type_ids": torch.zeros((batch_size, seq_len), dtype=torch.long).long().to(model.device),
        }
        times = []
        mems = []
        
        # Get current memory usage
        torch.cuda.reset_peak_memory_stats()
        current_mem = torch.cuda.max_memory_allocated()/1e9
        
        for i in range(20):
            t = time.time()
            out = model(dummy["input_ids"], attention_mask=dummy["attention_mask"], token_type_ids=dummy["token_type_ids"])
            times.append(time.time()-t)
            
            # loss = torch.nn.functional.cross_entropy(out.prediction_logits, torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long).to(model.device))
            # loss.backward()
            
            # Test memory usage during forward pass
            mems.append(torch.cuda.max_memory_allocated()/1e9 - current_mem)
            
            # Clear computation graph and clean memory
            del out
            model.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            
        return sum(times)/len(times), sum(mems)/len(mems)
    
    # Test different sequence lengths and batch sizes
    times = []
    mems = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"Testing batch size {batch_size} and sequence length {seq_len}")
            t, m = test(batch_size, seq_len)
            print(f"  Time: ", round(t, 2), "s")
            print(f"  Memory: ", round(m, 2), "GB")
            times.append(t)
            mems.append(m)
            
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    return times, mems


def main():
    use_amp = False
    train = True
    batch_sizes = [64]
    seq_lens = [8, 16, 32, 64, 128, 256, 512]
    models = {
        "Softmax": "models/redo_lr1e-4_SM/",
        "Cosine": "models/redo_lr1e-4_Cos_DivLearnLength/"
    }
    
    names, times, mems = [], [], []
    for model_name, model_path in models.items():
        t, m = test_model(model_path, batch_sizes, seq_lens, use_amp=use_amp, train=train)
        names.append(model_name)
        times.append(t)
        mems.append(m)
        
    print("Times: ", times)
    print("Memory: ", mems)
    
    # Plot times
    import matplotlib.pyplot as plt
    plt.plot(seq_lens, times[0], label=names[0], color="red")
    plt.plot(seq_lens, times[1], label=names[1], color="blue")
    plt.legend()
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (s)")
    plt.title("Time vs Sequence Length")
    plt.savefig("times.png")
    plt.show()
    plt.close()
    
    # Plot memory vs sequence length
    plt.plot(seq_lens, mems[0], label=names[0], color="red")
    plt.plot(seq_lens, mems[1], label=names[1], color="blue")
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory (GB)")
    plt.title("Memory vs Sequence Length")
    plt.legend()
    plt.savefig("memory.png")
    plt.show()
    plt.close()





if __name__ == "__main__":
    main()