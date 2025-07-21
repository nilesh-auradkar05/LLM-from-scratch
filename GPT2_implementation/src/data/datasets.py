import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm.auto import tqdm

class BookCorpusDataset(Dataset):
    def __init__(self, dataset, tokenizer, context_length, stride):
        """
        Args:
            book_corpus (dataset.Dataset): The raw dataset from Huggingface
            tokenizer: The tokenizer instance (e.g., from tiktoken)
            context_length (int): The length of the input sequence
            stride (int): The step size for sliding window.
        """
        self.input_ids = []
        self.target_ids = []
        
        # In distributed setting, only main process will show the progress bar
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        # 1. Concatenate all the text from the dataset and tokenize
        all_token_ids = []
        iterator = tqdm(dataset, desc="Tokenizing", disable=not is_main_process)
        for content in iterator:
            text = content.get("text")
            if text:
                all_token_ids.extend(tokenizer.encode_ordinary(text))
                
        # 2. Create chunks using sliding window
        chunk_iterator = tqdm(range(0, len(all_token_ids) - context_length, stride),
                              desc="Creating chunks", disable=not is_main_process)
        
        for i in chunk_iterator:
            input_chunk = all_token_ids[i : i+context_length]
            target_chunk = all_token_ids[i+1 : i+context_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
            