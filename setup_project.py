#!/usr/bin/env python3
# setup_project.py
# Copyright (c) 2025 Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)
# Description: Creates the initial project structure for Week 2 Two-Tower Model.
# Created: 2024-04-21

import os
import sys
from pathlib import Path

# --- Configuration ---
PROJECT_NAME = "MS MARCO Search Engine" # Or similar title
TEAM_NAME = "Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)"
COPYRIGHT_YEAR = "2025"
CREATION_DATE = "2025-04-21" # Assumed start date

# Basic project structure definition
DIRECTORIES = [
    "app",
    "data/msmarco",
    "docs",
    "logs",
    "models/two_tower",
    "notebooks",
    "scripts",
    "src",
    "src/two_tower",
    "utils",
]

# Files to create with basic content/headers
FILES_WITH_CONTENT = {
    ".gitignore": """\
# Python cache files
__pycache__/
*.py[cod]
*$py.class

# Environment files
.env
.venv/
venv/
env/
ENV/

# IDE folders
.vscode/
.idea/

# Data files (if large or specific - add data/* if needed)
# data/msmarco/*.tsv # Example - adjust if needed

# Model artifacts
models/

# Log files
logs/
*.log

# Notebook checkpoints
.ipynb_checkpoints/

# W&B local artifacts
wandb/

# OS-specific files
.DS_Store
Thumbs.db
""",
    "config.yaml": f"""\
# {PROJECT_NAME} - Configuration
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Created: {CREATION_DATE}

# --- General Paths ---
paths:
  # Input Data (MS MARCO v1.1 - Assumes download/placement)
  # Example using HuggingFace datasets structure might differ
  train_triples: "data/msmarco/train_triples.tsv" # Placeholder path
  val_triples: "data/msmarco/val_triples.tsv"   # Placeholder path
  corpus_embeddings: "models/two_tower/corpus_embeddings.pt" # For inference

  # Word2Vec Artifacts (Link to your Week 1 outputs)
  # Ensure these paths are correct relative to this project
  vocab_file: "../models/word2vec/text8_vocab_NWAll_MF5.json" # Example from W1
  pretrained_embeddings: "../models/word2vec/SkipGram_D128_W5_NWAll_MF5_E3_LR0.001_BS512/model_state.pth" # Example from W1

  # Two-Tower Model Output
  model_save_dir: "models/two_tower"
  log_dir: "logs"
  log_file_name: "two_tower_search.log"

# --- Word Embedding Parameters (from loaded model) ---
embeddings:
  freeze: True # Freeze pre-trained embeddings during tower training?
  # embed_dim will be inferred from loaded file usually

# --- Two Tower Model Hyperparameters ---
two_tower:
  model_type: "RNN" # Could be RNN, GRU, LSTM, BiLSTM etc.
  shared_document_encoder: True # Use same encoder for pos/neg docs?
  hidden_dim: 256 # Dimensionality of RNN hidden state / final encoding
  num_layers: 1 # Number of layers in RNN
  dropout: 0.1 # Dropout probability in RNN
  bidirectional: False # Use bidirectional RNN?

# --- Training Hyperparameters ---
training:
  epochs: 5
  batch_size: 128
  learning_rate: 0.0005 # Often lower for fine-tuning / triplet loss
  margin: 0.2 # Margin for Triplet Loss
  distance_metric: "cosine" # 'cosine' or 'euclidean'

# --- Logging Configuration ---
logging:
  log_level: "INFO"
  log_file_enabled: True
  log_console_enabled: True
  log_max_bytes: 10485760 # 10 MB
  log_backup_count: 5
""",
    "requirements.txt": """\
# Core ML Libraries
torch # Or torch==<version> torchvision torchaudio
pandas
numpy
scikit-learn # Often useful for metrics, splitting

# Utilities
PyYAML # For config.yaml
tqdm # Progress bars
wandb # Experiment tracking

# Specific Libraries (Add as needed)
# datasets # If using HuggingFace datasets library to load MS MARCO
# sentence-transformers # Might be useful for comparison or alternative models
""",
    "README.md": f"""\
# {PROJECT_NAME} 🚀

[![MLX Institute Logo](https://ml.institute/logo.png)](http://ml.institute)

A project by **Team {TEAM_NAME}** for Week 2 of the MLX Institute Intensive Program.

## Project Overview

This project implements a Two-Tower model for document retrieval (search). It uses pre-trained word embeddings (like those generated in Week 1) and trains two separate RNN-based towers (one for queries, one for documents) using Triplet Loss on the MS MARCO dataset. The goal is to learn dense vector representations for queries and documents such that relevant documents are closer to the query in the embedding space than irrelevant ones.

## Structure

(Refer to `docs/STRUCTURE.md` for a detailed breakdown - to be created by this script)

## Setup

1.  Clone the repository.
2.  Create and activate a Python virtual environment (`.venv`).
3.  Install dependencies: `pip install -r requirements.txt`
4.  Log in to Weights & Biases: `wandb login`
5.  Download MS MARCO v1.1 dataset (e.g., using HuggingFace `datasets`) and place relevant files (or configure paths in `config.yaml`).
6.  Ensure your pre-trained Word2Vec artifacts (vocab `.json`, model state `.pth`) from Week 1 are accessible and paths are correctly set in `config.yaml`.

## Usage

1.  **Configuration:** Modify `config.yaml` to set hyperparameters, paths, etc.
2.  **Training:** Run the training script:
    ```bash
    python scripts/train_two_tower.py
    ```
    (Monitor progress via console logs and the W&B dashboard link provided).
3.  **Evaluation:** (TODO: Implement evaluation script/notebook)
4.  **Inference:** (TODO: Implement inference script/notebook or API)

## Next Steps
*   Implement MS MARCO data loading and triplet generation (`src/two_tower/dataset.py`).
*   Implement Two-Tower model with RNN encoders (`src/two_tower/model.py`).
*   Implement Triplet Loss training loop (`src/two_tower/trainer.py`).
*   Refine `scripts/train_two_tower.py`.
*   Implement evaluation metrics (e.g., Recall@k, MRR).
""",
    "docs/STRUCTURE.md": f"""\
# {PROJECT_NAME} - Project Structure

This document outlines the directory structure for the Two-Tower search project.

```
.
├── config.yaml              # ⚙️ Central configuration file
├── data/                    # 📊 Input data
│   └── msmarco/             # MS MARCO specific data (e.g., downloaded files)
├── docs/                    # 📄 Project documentation
│   └── STRUCTURE.md         # This file
├── logs/                    # 📝 Runtime logs (Gitignored)
├── models/                  # 🧠 Saved model artifacts (Gitignored)
│   └── two_tower/           # Run-specific subdirs for this project
├── notebooks/               # 📓 Jupyter notebooks for exploration
├── scripts/                 # ▶️ Runnable scripts
│   ├── train_two_tower.py   # Script to orchestrate training
│   └── evaluate_two_tower.py # Placeholder for evaluation script
├── src/                     # 🐍 Core Python source code
│   ├── __init__.py
│   └── two_tower/           # Modules for the Two-Tower implementation
│       ├── __init__.py
│       ├── dataset.py       # MS MARCO data loading, triplet generation, PyTorch Dataset
│       ├── model.py         # Two-Tower model definition (Embeddings, RNN Towers)
│       └── trainer.py       # Training loop logic with Triplet Loss
├── utils/                   # 🛠️ Shared utility modules
│   ├── __init__.py
│   ├── device_setup.py      # CPU/MPS/CUDA device selection
│   ├── logging.py           # Logging setup
│   └── run_utils.py         # Config loading, artifact saving helpers
├── .gitignore               # Files/directories for Git to ignore
├── README.md                # 👋 High-level project overview
└── requirements.txt         # 📦 Python dependencies
```
*(Structure will evolve as development progresses)*
""",
    # --- Utils Files (Copied/Adapted from previous week) ---
    "utils/__init__.py": f"""\
# utils/__init__.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Created: {CREATION_DATE}

from .logging import logger
from .device_setup import get_device
from .run_utils import (
    format_num_words, load_config, save_losses, plot_losses
)

__all__ = [
    'logger', 'get_device', 'format_num_words', 'load_config',
    'save_losses', 'plot_losses'
]

logger.debug("Utils package initialized.")
""",
    "utils/logging.py": f"""\
# utils/logging.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Logging setup for the project.
# Created: {CREATION_DATE}

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

# --- Config ---
# Read from env or use defaults likely set by config.yaml loading later
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FILE_ENABLED = os.environ.get('LOG_FILE_ENABLED', 'true').lower() in ('true', 'yes', '1')
LOG_CONSOLE_ENABLED = os.environ.get('LOG_CONSOLE_ENABLED', 'true').lower() in ('true', 'yes', '1')
LOGS_DIR = os.environ.get('LOGS_DIR', 'logs')
LOG_FILE_NAME = os.environ.get('LOG_FILE_NAME', 'two_tower_search.log') # Default name
LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10*1024*1024)) # 10MB
LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
LOG_FORMAT = os.environ.get(
    'LOG_FORMAT',
    '%(asctime)s | %(name)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s'
)
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOGGER_NAME = "PerceptronSearch" # Specific logger name

logger = logging.getLogger(LOGGER_NAME)
_logging_initialized = False

def setup_logging(log_dir=LOGS_DIR, log_file=LOG_FILE_NAME): # Allow override
    '''Configures the project-specific logger.'''
    global _logging_initialized
    if _logging_initialized: return

    print(f"⚙️  Configuring {{LOGGER_NAME}} logging...")
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    print(f"  Logger '{{LOGGER_NAME}}' level set to: {{LOG_LEVEL}}")

    if logger.hasHandlers():
        print("  Clearing existing handlers...")
        for handler in logger.handlers[:]: logger.removeHandler(handler); handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if LOG_CONSOLE_ENABLED:
        ch = logging.StreamHandler(sys.stdout); ch.setLevel(level)
        ch.setFormatter(formatter); logger.addHandler(ch)
        print("  ✅ Console handler added.")

    if LOG_FILE_ENABLED:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
            with open(log_path, 'a', encoding='utf-8') as f: f.write("") # Check writability
            fh = RotatingFileHandler(log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8')
            fh.setLevel(level); fh.setFormatter(formatter); logger.addHandler(fh)
            print(f"  ✅ File handler added: {{log_path}}")
        except Exception as e: print(f"  ❌ ERROR setting up file log: {{e}}")

    if logger.hasHandlers(): logger.info("🎉 Logging system initialized!")
    else: print(f"⚠️ Warning: No handlers configured for {{LOGGER_NAME}}.")
    _logging_initialized = True

if not _logging_initialized: setup_logging()

if __name__ == "__main__": logger.info("Logging module test.")
""",
    "utils/device_setup.py": f"""\
# utils/device_setup.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Detects and sets the appropriate PyTorch device.
# Created: {CREATION_DATE}

import torch
import os
from .logging import logger

def get_device() -> torch.device:
    '''Checks for MPS, CUDA, or CPU and returns the device.'''
    selected_device = None
    logger.debug("⚙️  Checking for hardware accelerators...")
    if torch.backends.mps.is_available():
        selected_device = torch.device("mps")
        logger.info("✅ MPS device found. Using MPS.")
    elif torch.cuda.is_available():
        selected_device = torch.device("cuda")
        logger.info(f"✅ CUDA device found. Using CUDA ({{torch.cuda.get_device_name(0)}}).")
    else:
        selected_device = torch.device("cpu")
        logger.warning("⚠️ MPS/CUDA not available. Using CPU.")
    logger.info(f"✨ Selected compute device: {{selected_device.type.upper()}}")
    return selected_device

if __name__ == '__main__':
    logger.info("🚀 Running device setup check directly...")
    device = get_device()
    logger.info(f"🔍 Device object returned: {{device}}")
    try:
        x = torch.randn(2, 2, device=device)
        logger.info(f"✅ Test tensor created successfully on {{device}}.")
    except Exception as e: logger.error(f"❌ Failed test tensor on {{device}}: {{e}}")
""",
    "utils/run_utils.py": f"""\
# utils/run_utils.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Utility functions for running experiments.
# Created: {CREATION_DATE}

import os
import json
import yaml
import matplotlib.pyplot as plt
from typing import List
from .logging import logger

def format_num_words(num_words: int) -> str:
    '''Formats large numbers for filenames.'''
    if num_words == -1: return "All"
    if num_words >= 1_000_000: return f"{{num_words // 1_000_000}}M"
    if num_words >= 1_000: return f"{{num_words // 1_000}}k"
    return str(num_words)

def load_config(config_path: str = "config.yaml") -> dict | None:
    '''Loads configuration from a YAML file.'''
    logger.info(f"Loading configuration from: {{config_path}}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
        logger.info("✅ Configuration loaded successfully.")
        return config
    except Exception as e: logger.error(f"❌ Error loading config: {{e}}"); return None

def save_losses(losses: List[float], save_dir: str, filename: str = "training_losses.json") -> str | None:
    '''Saves epoch losses to a JSON file.'''
    if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
    loss_file = os.path.join(save_dir, filename)
    try:
        with open(loss_file, 'w', encoding='utf-8') as f: json.dump({{'epoch_losses': losses}}, f, indent=2)
        logger.info(f"📉 Training losses saved to: {{loss_file}}")
        return loss_file
    except Exception as e: logger.error(f"❌ Failed to save losses: {{e}}"); return None

def plot_losses(losses: List[float], save_dir: str, filename: str = "training_loss.png") -> str | None:
    '''Plots epoch losses and saves the plot.'''
    if not losses: return None
    if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
    plot_file = os.path.join(save_dir, filename)
    try:
        epochs = range(1, len(losses) + 1); plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o'); plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch'); plt.ylabel('Average Loss'); plt.xticks(epochs)
        plt.grid(True, ls='--'); plt.savefig(plot_file)
        logger.info(f"📈 Training loss plot saved to: {{plot_file}}"); plt.close()
        return plot_file
    except Exception as e: logger.error(f"❌ Failed to plot losses: {{e}}"); return None
""",
    # --- Src Files (Placeholders) ---
    "src/__init__.py": "# src/__init__.py: Makes 'src' a Python package.",
    "src/two_tower/__init__.py": f"""\
# src/two_tower/__init__.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Initializes the two_tower package.
# Created: {CREATION_DATE}

from .model import TwoTowerModel # Example import
# Add other relevant imports as needed
""",
    "src/two_tower/model.py": f"""\
# src/two_tower/model.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Defines the Two-Tower model architecture.
# Created: {CREATION_DATE}

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logger

class QueryEncoder(nn.Module):
    '''Encodes queries into dense vectors using an RNN (e.g., GRU/LSTM).'''
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False, rnn_type: str = 'GRU'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        rnn_class = getattr(nn, rnn_type.upper(), None)
        if rnn_class is None: raise ValueError(f"Invalid rnn_type: {{rnn_type}}")

        logger.info(f"Initializing Query Encoder ({{rnn_type.upper()}}), Hidden: {{hidden_dim}}, Layers: {{num_layers}}, Bidirectional: {{bidirectional}}")
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Important: assumes input shape (batch, seq_len, embed_dim)
            dropout=dropout if num_layers > 1 else 0, # Dropout only between layers
            bidirectional=bidirectional
        )

    def forward(self, query_embeds: torch.Tensor, query_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Encodes padded query embeddings.

        Args:
            query_embeds (torch.Tensor): Embeddings of shape (batch, seq_len, embed_dim).
            query_lengths (Optional[torch.Tensor]): Original lengths for packing (optional but recommended).

        Returns:
            torch.Tensor: Final query encoding (batch, hidden_dim * num_directions).
                          Usually the last hidden state.
        '''
        # PackedSequence handling recommended for variable lengths
        # packed_input = nn.utils.rnn.pack_padded_sequence(query_embeds, query_lengths.cpu(), batch_first=True, enforce_sorted=False)
        # packed_output, hidden = self.rnn(packed_input)
        # output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Simpler approach without packing (works if lengths are similar or using last state)
        output, hidden = self.rnn(query_embeds)

        # Extract final hidden state (depends on RNN type and layers/bidirectional)
        if isinstance(hidden, tuple): # LSTM: (h_n, c_n)
            hidden = hidden[0] # Take h_n
        # Handle layers and bidirectional
        # Shape: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            # Concatenate final forward and backward hidden states
            last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            last_hidden = hidden[-1,:,:] # Last layer's hidden state

        # Shape: (batch, hidden_dim * num_directions)
        return last_hidden


class DocumentEncoder(nn.Module):
    '''Encodes documents into dense vectors using an RNN (e.g., GRU/LSTM).'''
    # Similar structure to QueryEncoder, potentially identical if shared
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False, rnn_type: str = 'GRU'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        rnn_class = getattr(nn, rnn_type.upper(), None)
        if rnn_class is None: raise ValueError(f"Invalid rnn_type: {{rnn_type}}")

        logger.info(f"Initializing Document Encoder ({{rnn_type.upper()}}), Hidden: {{hidden_dim}}, Layers: {{num_layers}}, Bidirectional: {{bidirectional}}")
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def forward(self, doc_embeds: torch.Tensor, doc_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Encodes padded document embeddings.

        Args:
            doc_embeds (torch.Tensor): Embeddings of shape (batch, seq_len, embed_dim).
            doc_lengths (Optional[torch.Tensor]): Original lengths for packing.

        Returns:
            torch.Tensor: Final document encoding (batch, hidden_dim * num_directions).
        '''
        output, hidden = self.rnn(doc_embeds) # Simplified without packing for now
        if isinstance(hidden, tuple): hidden = hidden[0]
        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            last_hidden = hidden[-1,:,:]
        return last_hidden


class TwoTowerModel(nn.Module):
    '''The main Two-Tower model combining embedding, encoders, and potentially projection.'''
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, config: dict, pretrained_weights: Optional[torch.Tensor] = None):
        super().__init__()
        logger.info("Initializing TwoTowerModel...")
        w2v_config = config.get('embeddings', {{}})
        tower_config = config.get('two_tower', {{}})

        # --- Embedding Layer ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # Assume 0 is padding index
        if pretrained_weights is not None:
            logger.info("Loading pre-trained embedding weights.")
            self.embedding.weight.data.copy_(pretrained_weights)
        if w2v_config.get('freeze', True):
            logger.info("Freezing embedding layer weights.")
            self.embedding.weight.requires_grad = False
        else:
            logger.info("Fine-tuning embedding layer weights.")

        # --- Encoders ---
        encoder_args = {{
            "embedding_dim": embed_dim,
            "hidden_dim": tower_config.get('hidden_dim', 256),
            "num_layers": tower_config.get('num_layers', 1),
            "dropout": tower_config.get('dropout', 0.1),
            "bidirectional": tower_config.get('bidirectional', False),
            "rnn_type": tower_config.get('model_type', 'GRU')
        }}
        self.query_encoder = QueryEncoder(**encoder_args)

        if tower_config.get('shared_document_encoder', True):
            logger.info("Using shared encoder for documents.")
            self.doc_encoder = self.query_encoder
        else:
            logger.info("Using separate encoder for documents.")
            self.doc_encoder = DocumentEncoder(**encoder_args)

        # Optional: Projection head (if needed to map encodings to final space)
        # final_dim = encoder_args["hidden_dim"] * (2 if encoder_args["bidirectional"] else 1)
        # self.query_proj = nn.Linear(final_dim, final_dim) # Example: no dim change
        # self.doc_proj = nn.Linear(final_dim, final_dim)

    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        '''Encodes tokenized query IDs into a single vector.'''
        query_embeds = self.embedding(query_ids)
        query_encoding = self.query_encoder(query_embeds)
        # query_encoding = self.query_proj(query_encoding) # Apply projection if defined
        return query_encoding

    def encode_document(self, doc_ids: torch.Tensor) -> torch.Tensor:
        '''Encodes tokenized document IDs into a single vector.'''
        doc_embeds = self.embedding(doc_ids)
        doc_encoding = self.doc_encoder(doc_embeds)
        # doc_encoding = self.doc_proj(doc_encoding) # Apply projection if defined
        return doc_encoding

    def forward(self, query_ids: torch.Tensor, pos_doc_ids: torch.Tensor, neg_doc_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Processes a triplet through the towers.

        Args:
            query_ids (torch.Tensor): Batch of query token IDs.
            pos_doc_ids (torch.Tensor): Batch of positive document token IDs.
            neg_doc_ids (torch.Tensor): Batch of negative document token IDs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Query embeddings, Positive document embeddings, Negative document embeddings.
        '''
        q_embed = self.encode_query(query_ids)
        p_embed = self.encode_document(pos_doc_ids)
        n_embed = self.encode_document(neg_doc_ids)
        return q_embed, p_embed, n_embed

""",
    "src/two_tower/dataset.py": f"""\
# src/two_tower/dataset.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Data loading, preprocessing, triplet generation for MS MARCO.
# Created: {CREATION_DATE}

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
                                   tokenized IDs like {{'query': [ids], 'pos_doc': [ids], 'neg_doc': [ids]}}
        '''
        self.triplets = triplets
        logger.info(f"TripletDataset created with {{len(self.triplets)}} triplets.")

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
    max_lens = {{key: max(len(item[key]) for item in batch) for key in keys}}

    padded_batch = {{f"{{key}}_ids": [] for key in keys}}

    for item in batch:
        for key in keys:
            ids = item[key]
            padded_ids = ids + [padding_value] * (max_lens[key] - len(ids))
            padded_batch[f"{{key}}_ids"].append(padded_ids)

    # Convert lists to tensors
    for key in keys:
        padded_batch[f"{{key}}_ids"] = torch.tensor(padded_batch[f"{{key}}_ids"], dtype=torch.long)

    return padded_batch

# --- Placeholder functions ---
def load_msmarco_data(filepath: str) -> pd.DataFrame:
     logger.warning(f"Placeholder: Load MS MARCO data from {{filepath}}")
     # Example: return pd.read_csv(filepath, sep='\\t', header=None, names=['query', 'pos', 'neg'])
     return pd.DataFrame() # Return empty for now

def tokenize_text(text: str, vocab: Vocabulary) -> List[int]:
    logger.warning(f"Placeholder: Tokenize text: '{{text[:50]}}...'")
    # Simple split and lookup
    tokens = text.lower().split() # Basic tokenization
    return [vocab.get_index(token) for token in tokens]

def generate_triplets_from_data(df: pd.DataFrame, vocab: Vocabulary) -> List[Dict]:
     logger.warning("Placeholder: Generate indexed triplets from DataFrame")
     triplets = []
     # Example loop (replace with actual logic)
     # for _, row in df.iterrows():
     #     triplets.append({{
     #         'query': tokenize_text(row['query'], vocab),
     #         'pos_doc': tokenize_text(row['pos'], vocab),
     #         'neg_doc': tokenize_text(row['neg'], vocab)
     #     }})
     return triplets # Return empty list for now

""",
    "src/two_tower/trainer.py": f"""\
# src/two_tower/trainer.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Training loop for the Two-Tower model using Triplet Loss.
# Created: {CREATION_DATE}

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F # For distance functions
from tqdm import tqdm
from typing import List, Dict, Optional
from utils import logger
# Import specific model if needed, or just use nn.Module
# from .model import TwoTowerModel

# W&B Import Handling
try:
    import wandb
except ImportError: wandb = None

def calculate_distance(tensor1: torch.Tensor, tensor2: torch.Tensor, metric: str = 'cosine') -> torch.Tensor:
    '''Calculates distance between pairs of vectors.'''
    if metric == 'cosine':
        # Cosine similarity -> distance: 1 - similarity
        # Ensure tensors are normalized for stability? Optional.
        # sim = F.cosine_similarity(tensor1, tensor2, dim=1)
        # return 1.0 - sim
        # Or calculate distance directly (more stable for loss?)
        # Using negative similarity as a proxy for distance (minimize neg sim -> maximize sim)
         return - F.cosine_similarity(tensor1, tensor2, dim=1)
    elif metric == 'euclidean':
        return F.pairwise_distance(tensor1, tensor2, p=2)
    else:
        raise ValueError(f"Unknown distance metric: {{metric}}")


def train_epoch_two_tower(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    total_epochs: int,
    margin: float,
    distance_metric: str
) -> float:
    '''Trains the Two-Tower model for one epoch using Triplet Loss.'''
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    if num_batches == 0: 
        logger.warning("Empty dataloader, skipping epoch")
        return 0.0

    data_iterator = tqdm(dataloader, desc=f"Epoch {{epoch_num+1}}/{{total_epochs}}", leave=False, unit="batch")

    for batch_idx, batch in enumerate(data_iterator):
        # Assume collate_fn produces tensors on CPU, move them here
        query_ids = batch['query_ids'].to(device)
        pos_doc_ids = batch['pos_doc_ids'].to(device)
        neg_doc_ids = batch['neg_doc_ids'].to(device)

        optimizer.zero_grad()

        # Get embeddings from the model
        q_embed, p_embed, n_embed = model(query_ids, pos_doc_ids, neg_doc_ids)

        # Calculate distances
        # Note: If using cosine distance = 1 - sim, loss is max(0, d_pos - d_neg + margin)
        # If using distance = -sim, loss is max(0, (-sim_pos) - (-sim_neg) + margin) = max(0, sim_neg - sim_pos + margin)
        # Let's use distance = -similarity
        dist_pos = calculate_distance(q_embed, p_embed, distance_metric)
        dist_neg = calculate_distance(q_embed, n_embed, distance_metric)

        # Triplet Loss: max(0, dist_pos - dist_neg + margin)
        # Equivalently: max(0, margin + similarity_neg - similarity_pos) if using dist=-sim
        losses = F.relu(dist_pos - dist_neg + margin)
        loss = losses.mean() # Average loss over the batch

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        if batch_idx % 50 == 0:
            data_iterator.set_postfix(loss=f"{{batch_loss:.4f}}")

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss


def train_two_tower_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    # val_dataloader: Optional[DataLoader], # Add validation later
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict, # Pass config for training params
    wandb_run = None
) -> List[float]:
    '''Orchestrates training for the Two-Tower model.'''
    epochs = config.get('training', {{}}).get('epochs', 5)
    margin = config.get('training', {{}}).get('margin', 0.2)
    distance_metric = config.get('training', {{}}).get('distance_metric', 'cosine')
    model_save_dir = config.get('paths', {{}}).get('model_save_dir', 'models/two_tower')
    run_name = wandb_run.name if wandb_run else "two_tower_run" # Get run name for saving

    logger.info(f"🚀 Starting Two-Tower training ({{run_name}})")
    logger.info(f"  Epochs: {{epochs}}, Margin: {{margin}}, Distance: {{distance_metric}}")
    model.to(device)
    epoch_losses = []

    # if wandb_run and wandb: wandb.watch(model, log='all', log_freq=100)

    for epoch in range(epochs):
        avg_loss = train_epoch_two_tower(
            model, train_dataloader, optimizer, device, epoch, epochs, margin, distance_metric
        )
        logger.info(f"✅ Epoch {{epoch+1}}/{{epochs}} | Avg Train Loss: {{avg_loss:.4f}}")
        epoch_losses.append(avg_loss)

        # --- Log to W&B ---
        if wandb_run:
             log_data = {{"epoch": epoch + 1, "train_loss": avg_loss}}
             # Add val loss here if validation is implemented
             wandb_run.log(log_data)

        # --- Optional: Validation Step ---
        # if val_dataloader: evaluate(...)

    logger.info("🏁 Training finished.")
    # --- Save Model ---
    try:
        final_save_dir = os.path.join(model_save_dir, run_name) # Save in run-specific dir
        os.makedirs(final_save_dir, exist_ok=True)
        model_path = os.path.join(final_save_dir, "two_tower_final.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 Final model state saved to: {{model_path}}")
        # Log as W&B artifact later in main script
    except Exception as e:
        logger.error(f"❌ Failed to save final model: {{e}}")

    return epoch_losses

""",
    # --- Scripts (Placeholders) ---
    "scripts/train_two_tower.py": f"""\
# scripts/train_two_tower.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Main script to train the Two-Tower model for MS MARCO.
# Created: {CREATION_DATE}

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import wandb

# Adjust relative paths if necessary based on execution location
from utils import (
    logger, get_device, load_config, format_num_words,
    save_losses, plot_losses
)
# Note: Need to ensure word2vec components are accessible if loading embeddings
# This might require adding 'src' to sys.path if running from scripts/
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.word2vec.vocabulary import Vocabulary # Example import path
from src.two_tower.model import TwoTowerModel
from src.two_tower.dataset import (
    TripletDataset, collate_triplets,
    load_msmarco_data, generate_triplets_from_data # Placeholders
)
from src.two_tower.trainer import train_two_tower_model

def parse_train_args(config):
    '''Parse arguments specific to two-tower training.'''
    parser = argparse.ArgumentParser(description="Train Two-Tower Model.")
    # --- Add relevant args, using config for defaults ---
    paths = config.get('paths', {{}})
    training_cfg = config.get('training', {{}})
    w2v_paths = config.get('paths', {{}}) # Reuse paths for w2v artifacts

    parser.add_argument('--train-data', type=str, default=paths.get('train_triples'), help='Path to training data (e.g., triples TSV)')
    parser.add_argument('--model-save-dir', type=str, default=paths.get('model_save_dir'), help='Base directory to save models')
    parser.add_argument('--vocab-path', type=str, default=w2v_paths.get('vocab_file'), help='Path to pre-trained vocab JSON')
    parser.add_argument('--embedding-path', type=str, default=w2v_paths.get('pretrained_embeddings'), help='Path to pre-trained embedding state_dict (.pth)')

    parser.add_argument('--epochs', type=int, default=training_cfg.get('epochs'), help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=training_cfg.get('batch_size'), help='Training batch size')
    parser.add_argument('--lr', type=float, default=training_cfg.get('learning_rate'), help='Learning rate')

    # Add W&B args
    parser.add_argument('--wandb-project', type=str, default='perceptron-search-two-tower', help='W&B project')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--no-wandb', action='store_true', default=False, help='Disable W&B')

    args = parser.parse_args()
    logger.info("--- Effective Training Configuration ---")
    for arg, value in vars(args).items(): logger.info(f"  --{{arg.replace('_', '-'):<20}}: {{value}}")
    logger.info("------------------------------------")
    return args


def main():
    config = load_config()
    if config is None: return
    args = parse_train_args(config) # Parse specific args for this script

    # --- W&B Init ---
    run = None
    if not args.no_wandb:
        try:
            run_name = f"TwoTower_E{{args.epochs}}_LR{{args.lr}}_BS{{args.batch_size}}" # Example name
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_name, save_code=True)
            logger.info(f"📊 Initialized W&B run: {{run.name}} ({{run.get_url()}})")
        except Exception as e: logger.error(f"❌ Failed W&B init: {{e}}"); run = None
    else: logger.info("📊 W&B logging disabled.")

    logger.info(f"🚀 Starting Two-Tower Training...")
    device = get_device()

    # --- Load Pre-trained Vocab & Embeddings ---
    logger.info("--- Loading Pre-trained Artifacts ---")
    try:
        vocab = Vocabulary.load_vocab(args.vocab_path)
        logger.info(f"Loaded vocabulary ({{len(vocab)}} words)")
        if not os.path.exists(args.embedding_path): raise FileNotFoundError("Embedding file not found")
        # Load only the embedding weights, infer size later
        embedding_state = torch.load(args.embedding_path, map_location='cpu')
        # Determine the key for embeddings (might be 'in_embed.weight' or 'embeddings.weight')
        embed_key = 'in_embed.weight' if 'in_embed.weight' in embedding_state else 'embeddings.weight'
        if embed_key not in embedding_state: raise KeyError("Cannot find embedding weights in state dict")
        pretrained_weights = embedding_state[embed_key]
        embed_dim = pretrained_weights.shape[1] # Infer dimension
        logger.info(f"Loaded pre-trained embeddings. Shape: {{pretrained_weights.shape}}")
        # Update config/args with inferred embed_dim if needed
        config['embeddings']['embed_dim'] = embed_dim # Update loaded config dict
        args.embed_dim = embed_dim # Update args if needed elsewhere
    except Exception as e:
        logger.error(f"❌ Failed loading pre-trained artifacts: {{e}}", exc_info=True)
        if run: run.finish(exit_code=1)
        return

    # --- Load and Prepare MS MARCO Data ---
    # TODO: Replace placeholders with actual data loading & triplet generation
    logger.info("--- Preparing MS MARCO Data ---")
    # train_df = load_msmarco_data(args.train_data)
    # indexed_triplets = generate_triplets_from_data(train_df, vocab)
    # Using placeholder data for now:
    logger.warning("Using placeholder data for training!")
    indexed_triplets = [{{'query': [1,2,3], 'pos_doc': [4,5,6,7], 'neg_doc': [8,9,1,2]}} for _ in range(1000)] # Example
    if not indexed_triplets: logger.error("No training data."); return

    train_dataset = TripletDataset(indexed_triplets)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_triplets, # Use custom collate
        num_workers=0,
        pin_memory=(device.type != 'mps')
    )
    logger.info("DataLoader ready.")

    # --- Initialize Model & Optimizer ---
    logger.info("--- Initializing Two-Tower Model ---")
    model = TwoTowerModel(
        vocab_size=len(vocab),
        embed_dim=embed_dim, # Use inferred dim
        hidden_dim=config.get('two_tower', {{}}).get('hidden_dim', 256),
        config=config, # Pass full config
        pretrained_weights=pretrained_weights
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info("Model and optimizer ready.")

    # --- Train ---
    epoch_losses = train_two_tower_model(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        config=config, # Pass config for training params
        wandb_run=run
    )

    # --- Finalize ---
    run_save_dir = os.path.join(args.model_save_dir, run.name if run else "two_tower_local_run")
    loss_file = save_losses(epoch_losses, run_save_dir)
    plot_file = plot_losses(epoch_losses, run_save_dir)
    model_file = os.path.join(run_save_dir, "two_tower_final.pth") # Path where trainer saved

    if run:
        logger.info("☁️ Logging final artifacts to W&B...")
        try:
            final_artifact = wandb.Artifact(f"two_tower_final_{{run.id}}", type="model")
            final_artifact.add_file(model_file)
            if loss_file: final_artifact.add_file(loss_file)
            if plot_file: final_artifact.add_file(plot_file)
            # Add config.yaml to artifact for reproducibility
            final_artifact.add_file("config.yaml")
            run.log_artifact(final_artifact)
            logger.info("  Logged final model, results, and config artifact.")
        except Exception as e: logger.error(f"❌ Failed final W&B artifact logging: {{e}}")
        run.finish()
        logger.info("☁️ W&B run finished.")

    logger.info("✅ Two-Tower training process completed.")

if __name__ == "__main__":
    main()

""",
    "scripts/evaluate_two_tower.py": f"""\
# scripts/evaluate_two_tower.py
# Copyright (c) {COPYRIGHT_YEAR} {TEAM_NAME}
# Description: Script to evaluate the trained Two-Tower model.
# Created: {CREATION_DATE}

# TODO: Implement evaluation logic
# - Load model, vocab, precomputed document embeddings
# - Load evaluation queries and relevant documents (qrels)
# - Encode queries
# - Perform nearest neighbor search against document embeddings
# - Calculate metrics (Recall@k, MRR)

import argparse
from utils import logger

logger.info("Evaluation script placeholder.")

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Evaluate Two-Tower Model")
     # Add args for model path, data path, metrics etc.
     # args = parser.parse_args()
     logger.info("Evaluation script needs implementation.")
""",
    # --- Empty Init Files ---
    "app/__init__.py": "# app/__init__.py",
    "src/__init__.py": "# src/__init__.py",
    "src/two_tower/__init__.py": "# src/two_tower/__init__.py"
}

# Placeholder for directories that might just need __init__.py
TOUCH_FILES = {
    "app/__init__.py",
    "src/__init__.py",
    "src/two_tower/__init__.py",
    "utils/__init__.py"
}

# --- Script Logic ---
def create_file(filepath, content=''):
    '''Creates a file with given content, creating dirs if needed.'''
    filepath = Path(filepath)
    try:
        # Check if file already exists to avoid overwriting
        if filepath.exists():
            print(f"  ⚠️ File already exists: {{filepath}} (skipping)")
            return
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip() + "\n") # Add newline at end
        print(f"  📄 Created: {{filepath}}")
    except Exception as e:
        print(f"  ❌ ERROR creating {{filepath}}: {{e}}", file=sys.stderr)

def main_setup():
    '''Creates the project directory structure and files.'''
    project_root = Path(".") # Run from the project root directory
    print(f"Setting up project structure in: {project_root.resolve()}")

    # Create directories
    print("\nCreating directories...")
    for dirname in DIRECTORIES:
        dirpath = project_root / dirname
        try:
            dirpath.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Ensured directory: {dirpath}")
        except Exception as e:
            print(f"  ❌ ERROR creating directory {dirpath}: {{e}}", file=sys.stderr)
        # Create __init__.py if needed for package dirs
        if dirname.startswith("src/") or dirname.startswith("utils/") or dirname.startswith("app/"):
             init_file = dirpath / "__init__.py"
             if not init_file.exists() and str(init_file) not in FILES_WITH_CONTENT:
                  create_file(init_file, f"# {dirname}/__init__.py")


    # Create files with content
    print("\nCreating files...")
    for filepath_str, content in FILES_WITH_CONTENT.items():
        create_file(project_root / filepath_str, content)

    # Create empty __init__.py files if not covered above
    for filepath_str in TOUCH_FILES:
         filepath = project_root / filepath_str
         if not filepath.exists():
             create_file(filepath, f"# {filepath_str}")


    print("\n✨ Project structure setup complete! ✨")
    print("➡️ Next Steps:")
    print("1. Initialize Git: `git init && git add . && git commit -m 'Initial project structure'`")
    print("2. Create and activate virtual environment (e.g., `python -m venv .venv`)")
    print("3. Install dependencies: `pip install -r requirements.txt`")
    print("4. Download MS MARCO data into `data/msmarco/`")
    print("5. Update `config.yaml` with correct paths (especially Word2Vec artifacts)")
    print("6. Start implementing data loading in `src/two_tower/dataset.py`!")

if __name__ == "__main__":
    main_setup()