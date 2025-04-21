# Two Tower Search
# File: src/word2vec/__init__.py
# Copyright (c) 2025 Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)
# Description: Word2Vec model and utilities for training and evaluation.
# Created: 2025-04-21
# Updated: 2025-04-21

# Expose key components for easier importing if desired
from .model import CBOW
from .vocabulary import Vocabulary
from .dataset import create_cbow_pairs, CBOWDataset
from .trainer import train_model

__all__ = ['CBOW', 'Vocabulary', 'create_cbow_pairs', 'CBOWDataset', 'train_model']

# You can add a logger statement here if needed, but it's often kept minimal.
from utils import logger
logger.debug("Word2vec package initialized.")