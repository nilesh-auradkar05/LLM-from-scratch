import os
import torch
from loguru import logger
from datasets import load_dataset
import tiktoken
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import psutil
import wandb
from datetime import datetime

from src.model.modified_model_net import ModifiedGPT2ModelArch
from src.data.datasets import BookCorpusDataset
from src.utils.util import calculate_metrics


"""Main Script file to pre-train model on single GPU."""

def log_memory_usage(epoch):
    """Logs system and GPU memory usage to console and W&B."""
    # System RAM
    ram_stats = psutil.virtual_memory()
    ram_used_gb = ram_stats.used / (1024**3)
    ram_total_gb = ram_stats.total / (1024**3)
    
    # GPU VRAM
    vram_used_gb = 0
    vram_total_gb = 0
    if torch.cuda.is_available():
        vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        torch.cuda.reset_peak_memory_stats()

    logger.info(f"Memory Usage at Epoch {epoch}: RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB")
    wandb.log({
        "memory/epoch": epoch,
        "memory/ram_used_gb": ram_used_gb,
        "memory/vram_used_gb": vram_used_gb,
    })
    print(f"Memory Usage at Epoch {epoch}: RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB")

def main(cfg):
    """The main function for the single GPU training."""
    
    # 1. Setup
    logger.add("logs/pretrain_single_gpu_logs_{time}.log", format="{time} | {level} | {message}", level="INFO")

    # Wandb initialization
    wandb.init(
        project=config["wandb_config"]["project"],
        name=config["wandb_config"]["name"],
        config=config
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # 2. Model Inititalization
    model = ModifiedGPT2ModelArch(cfg["model_config"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with Total trainable parameters: {total_params:,}")
    wandb.watch(model, log="all", log_freq=100)

    # 2. Data Loading
    logger.info("Loading and preparing Dataset!")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    full_dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
    book_corpus = full_dataset.select(range(500_000))
    # book_corpus = full_dataset
    split_dataset = book_corpus.train_test_split(test_size=0.25, seed=47)
    train_set = split_dataset["train"]
    val_set = split_dataset["test"]
    
    train_data = BookCorpusDataset(dataset=train_set, tokenizer=tokenizer,
                                   context_length=cfg["model_config"]["context_length"],
                                   stride=cfg["model_config"]["context_length"])
    
    val_data = BookCorpusDataset(dataset=val_set, tokenizer=tokenizer,
                                 context_length=cfg["model_config"]["context_length"],
                                 stride=cfg["model_config"]["context_length"])
    
    # 3. Prepare Dataloaders
    train_loader = DataLoader(train_data, batch_size=cfg["train_config"]["batch_size"],
                              shuffle=True, num_workers=cfg["train_config"]["num_workers"],
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=cfg["train_config"]["batch_size"],
                            shuffle=False, num_workers=cfg["train_config"]["num_workers"],
                            pin_memory=True)
    logger.info("Dataloaders created!")
    
    # 4. Optimizer and Scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train_config"]["learning_rate"],
                                  weight_decay=cfg["train_config"]["weight_decay"])
    num_training_steps = len(train_loader) * cfg["train_config"]["num_epochs"]
    warmup_steps = num_training_steps * 0.1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # 5. Training loop
    logger.info("Starting training!")
    for epoch in range(1, cfg["train_config"]["num_epochs"]+1):
        # Train Phase
        model.train()
        train_loss, train_acc, train_perplexity = 0.0, 0.0, 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch}")
        for step, (input_batch, target_batch) in enumerate(train_progress_bar):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(input_batch)
            loss, acc, perplexity = calculate_metrics(logits, target_batch)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            # log metrics to wandb
            wandb.log({
                "train/step_loss": loss.item(),
                "train/step_accuracy": acc,
                'train/learning_rate': scheduler.get_last_lr()[0],
            })
            
            train_loss += loss.item()
            train_acc += acc
            train_perplexity += perplexity.item()
            
            train_progress_bar.postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{acc:.2f}",
                "perplexity score": f"{perplexity.item():.4f}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_train_perplexity = train_perplexity / len(train_perplexity)
        
        # Validation Phase
        model.eval()
        val_loss, val_acc, val_perplexity = 0.0, 0.0, 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Validation epoch {epoch}")
        with torch.no_grad():
            for input_batch, target_batch in val_progress_bar:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                logits = model(input_batch)
                loss, acc, perplexity = calculate_metrics(logits, target_batch)
                
                val_loss += loss.item()
                val_acc += acc
                val_perplexity += perplexity.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_perplexity = val_perplexity / len(val_perplexity)

        # Logging and checkpointing
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/accuracy": avg_train_acc,
            "train/perplexity": avg_train_perplexity,
            "val/loss": avg_val_loss,
            "val/accuracy": avg_val_acc,
            "val/perplexity": avg_val_perplexity,
        })

        log_memory_usage(epoch)
        
        checkpoint_path = f"{cfg["experiment_path"]}/epoch_{epoch}_vloss_{avg_val_loss:.2f}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path)
        logger.info(f"Model Checkpoint saved at {checkpoint_path}")

    wandb.finish()
        
if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    config = {
        "model_config": {
            "vocab_size": 50257,
            "context_length": 1024,
            "embedding_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "drop_rate": 0.1,
            "qkv_bias": False,
        },
        "train_config": {
            "batch_size": 2,
            "num_workers": 2,
            "learning_rate": 5e-5,
            "num_epochs": 10,
        },
        "experiment_path": "experiments/pretrain_runs/run_single_gpu/checkpoints",
        "wandb_config": {
            "project": "experimental-gpt2-modified-arch-pretraining",
            "name": "single-gpu-script-run-{timestamp}",
        }
    }
    
    os.makedirs(config["experiment_path"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    main(config)
    print("Training completed on single GPU!")