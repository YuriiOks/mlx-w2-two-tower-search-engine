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

class QueryEncoder(nn.Module):
    '''Encodes queries into dense vectors using an RNN (e.g., GRU/LSTM).'''
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False, rnn_type: str = 'GRU'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        rnn_class = getattr(nn, rnn_type.upper(), None)
        if rnn_class is None: raise ValueError(f"Invalid rnn_type: {rnn_type}")

        logger.info(f"Initializing Query Encoder ({rnn_type.upper()}), Hidden: {hidden_dim}, Layers: {num_layers}, Bidirectional: {bidirectional}")
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
        if rnn_class is None: raise ValueError(f"Invalid rnn_type: {rnn_type}")

        logger.info(f"Initializing Document Encoder ({rnn_type.upper()}), Hidden: {hidden_dim}, Layers: {num_layers}, Bidirectional: {bidirectional}")
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
            "dropout": tower_config.get('dropout', 0.1),
            "bidirectional": tower_config.get('bidirectional', False),
            "rnn_type": tower_config.get('model_type', 'GRU')
        }
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
