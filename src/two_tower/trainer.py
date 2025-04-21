# Two Tower Search
# File: src/two_tower/trainer.py
# Copyright (c) 2025 Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)
# Description: Detects and sets the appropriate PyTorch device.
# Created: 2025-04-21
# Updated: 2025-04-21

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F # For distance functions
from tqdm import tqdm
from typing import List, Dict, Optional
from utils import logger
import os
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
        raise ValueError(f"Unknown distance metric: {metric}")


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

    data_iterator = tqdm(dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs}", leave=False, unit="batch")

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
            data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

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
    epochs = config.get('training', {}).get('epochs', 5)
    margin = config.get('training', {}).get('margin', 0.2)
    distance_metric = config.get('training', {}).get('distance_metric', 'cosine')
    model_save_dir = config.get('paths', {}).get('model_save_dir', 'models/two_tower')
    run_name = wandb_run.name if wandb_run else "two_tower_run" # Get run name for saving

    logger.info(f"🚀 Starting Two-Tower training ({run_name})")
    logger.info(f"  Epochs: {epochs}, Margin: {margin}, Distance: {distance_metric}")
    model.to(device)
    epoch_losses = []

    # if wandb_run and wandb: wandb.watch(model, log='all', log_freq=100)

    for epoch in range(epochs):
        avg_loss = train_epoch_two_tower(
            model, train_dataloader, optimizer, device, epoch, epochs, margin, distance_metric
        )
        logger.info(f"✅ Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

        # --- Log to W&B ---
        if wandb_run:
             log_data = {"epoch": epoch + 1, "train_loss": avg_loss}
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
        logger.info(f"💾 Final model state saved to: {model_path}")
        # Log as W&B artifact later in main script
    except Exception as e:
        logger.error(f"❌ Failed to save final model: {e}")

    return epoch_losses
