import os
import torch
from torch.utils.data import DataLoader
from loguru import logger
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup
import tiktoken
import wandb
import psutil
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger

from src.model.modified_model_net import ModifiedGPT2ModelArch
from src.data.datasets import BookCorpusDataset
from src.utils.util import calculate_metrics

"""This script is used to train the model on multiple GPUs using pytorch lightning using DDP"""

class MemoryUsageLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # Log system memory usage
        ram_stats = psutil.virtual_memory()
        ram_used_gb = ram_stats.used / (1024 ** 3)
        ram_total_gb = ram_stats.total / (1024 ** 3)
        
        # Log GPU memory usage
        vram_used_gb = 0
        vram_total_gb = 0
        if torch.cuda.is_available():
            # current device stats
            device_idx = trainer.local_rank
            vram_used_gb = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            vram_total_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            # Reset peak memory stats at the end of the epoch to track per epoch peak
            torch.cuda.reset_peak_memory_stats(device_idx)
            
        metrics = {
            "memory/ram_used_gb": ram_used_gb,
            "memory/ram_total_gb": ram_total_gb,
            "memory/vram_used_gb": vram_used_gb,
            "memory/vram_total_gb": vram_total_gb,
        }
        
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
        
        if trainer.is_global_zero:
            logger.info(f"Memory Usage at Epoch {trainer.current_epoch} | RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB")
            print(f"Memory Usage at Epoch {trainer.current_epoch} | RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB")

# 1. Lightning Data Module
class BookCorpusDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.context_length = self.cfg["model_config"]["context_length"]
        self.batch_size = self.cfg["train_config"]["batch_size"]
        self.num_workers = self.cfg["train_config"]["num_workers"]

        print(f"context_length: {self.context_length}, batch_size: {self.batch_size}, num_workers: {self.num_workers}")
        
    def setup(self, stage=None):
        full_dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
        book_corpus = full_dataset.select(range(500_000))
        # book_corpus = full_dataset
        split_dataset = book_corpus.train_test_split(test_size=0.25)
        print(split_dataset)
        try:
            train_set = split_dataset["train"]
            val_set = split_dataset["test"]
        except Exception as e:
            print(e)
            print(split_dataset)

        if stage == "fit" or stage is None:
        
            self.train_set = BookCorpusDataset(train_set,
                                               self.tokenizer,
                                               context_length=self.context_length,
                                               stride=self.context_length)
            self.val_set = BookCorpusDataset(val_set,
                                             self.tokenizer,
                                             context_length=self.context_length,
                                             stride=self.context_length)
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True, pin_memory=True, drop_last=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False, pin_memory=True, drop_last=False)
        
# 2. Lightning Module
class ModifiedGPT2LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = ModifiedGPT2ModelArch(cfg["model_config"])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss, acc, perplexity = calculate_metrics(logits, target_batch)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_accuracy", acc, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_perplexity", perplexity, on_epoch=True, logger=True, sync_dist=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss, acc, perplexity = calculate_metrics(logits, target_batch)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_accuracy", acc, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity, on_epoch=True, logger=True, sync_dist=True)

        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.train_config["learning_rate"],
                                      weight_decay=self.hparams.train_config["weight_decay"])
        num_training_steps = self.trainer.estimated_stepping_batches
        warmup_steps = num_training_steps * 0.1
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
# 3. Main Training Function
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    config = {
        "model_config": {
            "vocab_size": 50257,
            "context_length": 512,
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
            "num_epochs": 1,
            "warmup_steps": 700,
            "weight_decay": 0.1,
        },
        "experiment_path": "experiments/pretrain_runs/run_lightning/checkpoints",
        "wandb_config": {
            "project": "experimental-gpt2-modified-arch-pretraining",
            "name": "lightning-run-{timestamp}",
        }
    }
    
    os.makedirs(config["experiment_path"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 3. Initialize modules and trainer
    data_module = BookCorpusDataModule(config)
    model_module = ModifiedGPT2LightningModule(config)
    
    # Logger and Checkpoint Config
    wandb_logger = WandbLogger(
        name=config["wandb_config"]["name"],
        project=config["wandb_config"]["project"],
        log_model="all",
        config=config,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["experiment_path"],
        filename="experimentation-gpt2-modified-arch-epoch{epoch:02d}-vloss{val_loss:.2f}",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
    )
    
    memory_logger_callback = MemoryUsageLogger()
    
    # 4. Run Training
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        max_epochs=config["train_config"]["num_epochs"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, memory_logger_callback]
    )
    
    trainer.fit(model_module, data_module)
    
    # 5. Finish WandB
    if trainer.is_global_zero:
        wandb.finish()
    
    print("Training completed on multiple GPUs using pytorch lightning!")
    