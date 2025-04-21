# Two Tower Search
# File: src/two_tower/dataset.py
# Copyright (c) 2025 Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)
# Description: Detects and sets the appropriate PyTorch device.
# Created: 2025-04-21
# Updated: 2025-04-21

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import pandas as pd # For reading potential TSV files
from utils import logger
# Assuming word2vec vocab is used for tokenization initially
from src.word2vec.vocabulary import Vocabulary # Adjust path if needed

# TODO: Implement functions to:
# 1. Load MS MARCO data (e.g., from HF datasets or downloaded files)
# 2. Tokenize queries and passages using the loaded Word2Vec vocabulary
# 3. Generate triplets (query_idx, pos_doc_idx, neg_doc_idx) - Requires negative sampling strategy
# 4. Handle padding/truncation of sequences

class TripletDataset(Dataset):
    '''PyTorch Dataset for (query, positive_doc, negative_doc) triplets.'''
    def __init__(self, triplets: List[Dict]):
        '''
        Args:
            triplets (List[Dict]): List of dictionaries, each containing
                                   tokenized IDs like {'query': [ids], 'pos_doc': [ids], 'neg_doc': [ids]}
        '''
        self.triplets = triplets
        logger.info(f"TripletDataset created with {len(self.triplets)} triplets.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        # Return the dictionary containing the triplet IDs
        return self.triplets[idx]

def collate_triplets(batch: List[Dict[str, List[int]]], padding_value: int = 0) -> Dict[str, torch.Tensor]:
    '''
    Collates a batch of triplets, padding sequences to the max length in the batch.

    Args:
        batch (List[Dict[str, List[int]]]): A list of triplet dictionaries.
        padding_value (int): The index used for padding (usually 0 for <UNK> or a dedicated <PAD> token).

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing padded tensors for
                                 'query_ids', 'pos_doc_ids', 'neg_doc_ids'.
    '''
    keys = ['query', 'pos_doc', 'neg_doc']
    max_lens = {key: max(len(item[key]) for item in batch) for key in keys}

    padded_batch = {f"{key}_ids": [] for key in keys}

    for item in batch:
        for key in keys:
            ids = item[key]
            padded_ids = ids + [padding_value] * (max_lens[key] - len(ids))
            padded_batch[f"{key}_ids"].append(padded_ids)

    # Convert lists to tensors
    for key in keys:
        padded_batch[f"{key}_ids"] = torch.tensor(padded_batch[f"{key}_ids"], dtype=torch.long)

    return padded_batch

# --- Placeholder functions ---
def load_msmarco_data(filepath: str) -> pd.DataFrame:
     logger.warning(f"Placeholder: Load MS MARCO data from {filepath}")
     # Example: return pd.read_csv(filepath, sep='\t', header=None, names=['query', 'pos', 'neg'])
     return pd.DataFrame() # Return empty for now

def tokenize_text(text: str, vocab: Vocabulary) -> List[int]:
    logger.warning(f"Placeholder: Tokenize text: '{text[:50]}...'")
    # Simple split and lookup
    tokens = text.lower().split() # Basic tokenization
    return [vocab.get_index(token) for token in tokens]

def generate_triplets_from_data(df: pd.DataFrame, vocab: Vocabulary) -> List[Dict]:
     logger.warning("Placeholder: Generate indexed triplets from DataFrame")
     triplets = []
     # Example loop (replace with actual logic)
     # for _, row in df.iterrows():
     #     triplets.append({
     #         'query': tokenize_text(row['query'], vocab),
     #         'pos_doc': tokenize_text(row['pos'], vocab),
     #         'neg_doc': tokenize_text(row['neg'], vocab)
     #     })
     return triplets # Return empty list for now
