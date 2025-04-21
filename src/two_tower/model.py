# Two Tower Search
# File: src/two_tower/model.py
# Copyright (c) 2025 Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)
# Description: Detects and sets the appropriate PyTorch device.
# Created: 2025-04-21
# Updated: 2025-04-21

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logger
from typing import Optional, Tuple

class EncoderAveragePooling(nn.Module):
    '''Encodes queries into dense vectors using a linear layer and then average pooling.'''
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        logger.info(f"Initializing Query Encoder, Hidden: {hidden_dim}, Layers: {num_layers}")
        
        # Create a list of layers
        layers = []
        input_dim = embedding_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)

    def forward(self, query_embeds: torch.Tensor) -> torch.Tensor:
        output = self.network(query_embeds)
        output = torch.mean(output, dim=1)
        return output


class TwoTowerModelAveragePooling(nn.Module):
    '''The main Two-Tower model combining embedding, encoders, and potentially projection.'''
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, config: dict, pretrained_weights: Optional[torch.Tensor] = None):
        super().__init__()
        logger.info("Initializing TwoTowerModel...")
        w2v_config = config.get('embeddings', {})
        tower_config = config.get('two_tower', {})

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
        encoder_args = {
            "embedding_dim": embed_dim,
            "hidden_dim": tower_config.get('hidden_dim', 256),
            "num_layers": tower_config.get('num_layers', 1),
        }
        self.query_encoder = EncoderAveragePooling(**encoder_args)
        self.doc_encoder = EncoderAveragePooling(**encoder_args)


    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        '''Encodes tokenized query IDs into a single vector.'''
        query_embeds = self.embedding(query_ids)
        query_encoding = self.query_encoder(query_embeds)
        return query_encoding

    def encode_document(self, doc_ids: torch.Tensor) -> torch.Tensor:
        '''Encodes tokenized document IDs into a single vector.'''
        doc_embeds = self.embedding(doc_ids)
        doc_encoding = self.doc_encoder(doc_embeds)
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
