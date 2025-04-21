# Two Tower Search
# File: scripts/train_two_tower.py
# Copyright (c) 2025 Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)
# Description: Main script to train the Two-Tower model for MS MARCO.
# Created: 2025-04-21
# Updated: 2025-04-21

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import wandb
import sys
import functools

script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root: Dropout_Disco/)
project_root = os.path.dirname(script_dir)
# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    print(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)

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
    load_msmarco_hf,             # Function to load HF dataset
    generate_triplets_from_dataset, # Function to generate triplets
    TripletDataset,              # PyTorch Dataset class
    collate_triplets             # Function for padding batches
)

from src.two_tower.trainer import train_two_tower_model

def parse_train_args(config):
    '''Parse arguments specific to two-tower training.'''
    parser = argparse.ArgumentParser(description="Train Two-Tower Model.")
    # --- Read defaults from CORRECT config sections ---
    paths = config.get('paths', {})
    # Use 'two_tower_training' section for these defaults
    training_cfg = config.get('two_tower_training', {}) # <--- CHANGE HERE

    parser.add_argument('--train-data', type=str, default=paths.get('train_triples'), help='Path to training data (e.g., triples TSV)')
    # Default to the specific two_tower save dir from config
    parser.add_argument('--model-save-dir', type=str, default=paths.get('two_tower_model_save_dir'), help='Base directory to save models') # <--- CHANGE HERE
    parser.add_argument('--vocab-path', type=str, default=paths.get('vocab_file'), help='Path to pre-trained vocab JSON')
    # Default to the specific embedding path from config
    parser.add_argument('--embedding-path', type=str, default=paths.get('pretrained_embeddings'), help='Path to pre-trained embedding state_dict (.pth)') # <--- CHANGE HERE

    # Use training_cfg read from 'two_tower_training'
    parser.add_argument('--epochs', type=int, default=training_cfg.get('epochs'), help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=training_cfg.get('batch_size'), help='Training batch size') # <--- Should now read from training_cfg
    parser.add_argument('--lr', type=float, default=training_cfg.get('learning_rate'), help='Learning rate') # <--- Should now read from training_cfg

    # Add W&B args
    parser.add_argument('--wandb-project', type=str, default='perceptron-search-two-tower', help='W&B project')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity') # Keep default=None if not always needed
    parser.add_argument('--no-wandb', action='store_true', default=False, help='Disable W&B')

    args = parser.parse_args()
    logger.info("--- Effective Training Configuration ---")
    # Check the parsed args carefully after changes
    for arg, value in vars(args).items(): logger.info(f"  --{arg.replace('_', '-'):<20}: {value}")
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
            run_name = f"TwoTower_E{args.epochs}_LR{args.lr}_BS{args.batch_size}" # Example name
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_name, save_code=True)
            logger.info(f"📊 Initialized W&B run: {run.name} ({run.get_url()})")
        except Exception as e: logger.error(f"❌ Failed W&B init: {e}"); run = None
    else: logger.info("📊 W&B logging disabled.")

    logger.info(f"🚀 Starting Two-Tower Training...")
    device = get_device()

    # --- Load Pre-trained Vocab & Embeddings ---
    logger.info("--- Loading Pre-trained Artifacts ---")
    try:
        vocab = Vocabulary.load_vocab(args.vocab_path)
        logger.info(f"Loaded vocabulary ({len(vocab)} words)")
        if not os.path.exists(args.embedding_path): raise FileNotFoundError("Embedding file not found")
        # Load only the embedding weights, infer size later
        embedding_state = torch.load(args.embedding_path, map_location='cpu')
        # Determine the key for embeddings (might be 'in_embed.weight' or 'embeddings.weight')
        embed_key = 'in_embed.weight' if 'in_embed.weight' in embedding_state else 'embeddings.weight'
        if embed_key not in embedding_state: raise KeyError("Cannot find embedding weights in state dict")
        pretrained_weights = embedding_state[embed_key]
        embed_dim = pretrained_weights.shape[1] # Infer dimension
        logger.info(f"Loaded pre-trained embeddings. Shape: {pretrained_weights.shape}")
        # Update config/args with inferred embed_dim if needed
        config['embeddings']['embed_dim'] = embed_dim # Update loaded config dict
        args.embed_dim = embed_dim # Update args if needed elsewhere
    except Exception as e:
        logger.error(f"❌ Failed loading pre-trained artifacts: {e}", exc_info=True)
        if run: run.finish(exit_code=1)
        return

    # --- Load and Prepare MS MARCO Data ---
    logger.info("--- Preparing MS MARCO Data ---")
    try:
        # 1. Load HF Dataset
        # Use 'train' split for training. Set streaming=False for standard loading.
        train_dataset_hf = load_msmarco_hf(split='train', streaming=False)
        if train_dataset_hf is None:
            raise RuntimeError("Failed to load MS MARCO Hugging Face dataset.")

        # 2. Generate Tokenized Triplets
        # Get max lengths from config, provide defaults if missing
        tower_config = config.get('two_tower', {})
        max_q_len = tower_config.get('max_query_len', 64)  # Default 64
        max_d_len = tower_config.get('max_doc_len', 256) # Default 256

        # Generate triplets from the full training dataset
        # Set max_triplets=None to process everything (or a number for testing)
        logger.info("Generating tokenized triplets from the loaded dataset...")
        tokenized_triplets = generate_triplets_from_dataset(
            train_dataset_hf,
            vocab,           # Pass the loaded vocabulary object
            max_q_len,
            max_d_len,
            max_triplets=None # Process full dataset for real training
            # max_triplets=10000 # Or set a limit for a faster test run initially
        )

        if not tokenized_triplets:
            raise RuntimeError("No tokenized triplets were generated. Check data or vocab.")

        # 3. Create PyTorch Dataset
        train_dataset = TripletDataset(tokenized_triplets)

        # 4. Determine Padding Index
        # Use the UNK index from the vocabulary as the padding value
        padding_idx = vocab.unk_index if hasattr(vocab, 'unk_index') else 0
        logger.info(f"Using padding index: {padding_idx}")


        # 5. Create DataLoader
        # Use number of CPU cores for num_workers, adjust if needed
        num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
        collate_partial = functools.partial(collate_triplets, padding_value=padding_idx)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, # From command line args/config
            shuffle=True,               # Shuffle training data
            collate_fn=collate_partial, # <-- Use the partial object
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False # pin_memory useful for GPU
        )
        logger.info(f"✅ MS MARCO DataLoader ready ({len(train_dataset)} triplets).")

    except Exception as e:
        logger.error(f"❌ Failed during data preparation: {e}", exc_info=True)
        if run: run.finish(exit_code=1) # Finish W&B run if it started
        return # Stop execution

    # --- Initialize Model & Optimizer ---
    logger.info("--- Initializing Two-Tower Model ---")
    model = TwoTowerModel(
        vocab_size=len(vocab),
        embed_dim=embed_dim, # Use inferred dim
        hidden_dim=config.get('two_tower', {}).get('hidden_dim', 256),
        config=config, # Pass full config
        pretrained_weights=pretrained_weights
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info("Model and optimizer ready.")

    # --- Train ---
    epoch_losses = train_two_tower_model(
        model=model,
        train_dataloader=train_dataloader, # Now uses the real dataloader!
        optimizer=optimizer,
        device=device,
        config=config,
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
            final_artifact = wandb.Artifact(f"two_tower_final_{run.id}", type="model")
            final_artifact.add_file(model_file)
            if loss_file: final_artifact.add_file(loss_file)
            if plot_file: final_artifact.add_file(plot_file)
            # Add config.yaml to artifact for reproducibility
            final_artifact.add_file("config.yaml")
            run.log_artifact(final_artifact)
            logger.info("  Logged final model, results, and config artifact.")
        except Exception as e: logger.error(f"❌ Failed final W&B artifact logging: {e}")
        run.finish()
        logger.info("☁️ W&B run finished.")

    logger.info("✅ Two-Tower training process completed.")

if __name__ == "__main__":
    main()
