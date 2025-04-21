# Two Tower Search
# File: src/two_tower/dataset.py
# Copyright (c) 2025 Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)
# Description: Data loading, preprocessing, triplet generation for MS MARCO.
# Created: 2025-04-21
# Updated: 2025-04-22 (Handles HF dataset structure and includes loading)

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import pandas as pd # Keep for potential future use
from datasets import (
    load_dataset, 
    Dataset as HFDataset, 
    ClassLabel, 
    IterableDataset
)
from tqdm import tqdm
import random
import os
import sys

# --- Add project root to sys.path for imports ---
# This allows running this script directly for testing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) # Go up two levels
if project_root not in sys.path:
    print(f"[Dataset Script] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)

# Assuming utils and word2vec are importable from the project root
from utils import logger, load_config # Import logger and config loader
from src.word2vec.vocabulary import Vocabulary # Adjust path if needed

# --- Constants ---
MSMARCO_DATASET_ID = "microsoft/ms_marco"
MSMARCO_VERSION = "v1.1"

# --- Data Loading Function ---
def load_msmarco_hf(split: str = 'train', 
                   streaming: bool = False) -> Optional[HFDataset]:
    """
    Loads a specific split of the MS MARCO v1.1 dataset using Hugging Face.

    Args:
        split (str): Dataset split to load ('train', 'validation', 'test').
        streaming (bool): Whether to load the dataset in streaming mode.

    Returns:
        Optional[HFDataset]: The loaded HF dataset object, or None if failed.
    """
    logger.info(f"📚 Attempting to load MS MARCO dataset: "
               f"ID='{MSMARCO_DATASET_ID}', "
               f"Version='{MSMARCO_VERSION}', "
               f"Split='{split}', Streaming={streaming}...")
    try:
        dataset = load_dataset(
            MSMARCO_DATASET_ID,
            MSMARCO_VERSION,
            split=split,
            streaming=streaming
            # cache_dir="path/to/cache" # Optional: specify cache directory
        )
        logger.info(f"✅ Successfully loaded dataset split '{split}'.")

        # Log features only once if possible
        if hasattr(dataset, 'features'):
            logger.info(f"📊 Features: {dataset.features}")
        else:
            logger.info("📊 (Features cannot be listed in streaming mode)")

        # Print basic info about the loaded data (if not streaming)
        if not streaming:
            try:
                logger.info(f"📊 Number of examples: {len(dataset)}")
                # Check query_type distribution if it's a ClassLabel
                if ('query_type' in dataset.features and 
                    isinstance(dataset.features['query_type'], ClassLabel)):
                    logger.info(f"📊 Query Types: {dataset.features['query_type'].names}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get detailed info: {e}")

        return dataset

    except Exception as e:
        logger.error(f"❌ Failed to load MS MARCO dataset split '{split}': {e}", 
                    exc_info=True)
        logger.error("❌ Please ensure 'datasets' library is installed")
        logger.error("❌ Check network connection and dataset availability.")
        return None

# --- Tokenization Helper ---
def tokenize_text(text: str, vocab: Vocabulary, max_len: int) -> List[int]:
    """
    Tokenizes text using the provided vocabulary and truncates.
    Applies basic lowercasing.
    """
    if not isinstance(text, str): # Basic type check
        logger.warning(f"⚠️ Invalid input to tokenize_text, expected string, "
                      f"got {type(text)}. Returning empty list.")
        return []
    # Basic preprocessing (match Word2Vec if possible)
    tokens = text.lower().split()
    # Map to indices, using UNK for unknown words
    indexed_tokens = [vocab.get_index(token) for token in tokens]
    # Truncate if longer than max_len
    return indexed_tokens[:max_len]

# --- Triplet Generation Logic ---
def generate_triplets_from_dataset(
    hf_dataset: HFDataset,
    vocab: Vocabulary,
    max_query_len: int,
    max_doc_len: int,
    max_triplets: Optional[int] = None # Optional limit for debugging
) -> List[Dict[str, List[int]]]:
    """
    Generates tokenized (query, positive_doc, negative_doc) triplets
    from the MS MARCO Hugging Face dataset. Handles streaming datasets.

    Args:
        hf_dataset: The loaded Hugging Face dataset (e.g., train split).
        vocab: The pre-loaded Vocabulary object.
        max_query_len: Maximum length for tokenized queries.
        max_doc_len: Maximum length for tokenized documents.
        max_triplets: Max number of triplets to generate (for testing).

    Returns:
        List of dictionaries, each with tokenized 'query', 'pos_doc', 'neg_doc'.
    """
    # Determine if dataset is iterable (streaming) or indexable
    is_iterable = isinstance(hf_dataset, IterableDataset)

    if is_iterable:
        logger.info(f"⚙️ Generating triplets from streaming MS MARCO examples...")
    else:
        logger.info(f"⚙️ Generating triplets from {len(hf_dataset):,} examples...")

    tokenized_triplets = []
    skipped_count = 0
    processed_count = 0

    # Use tqdm only if not streaming, as streaming doesn't have a fixed length
    data_iterator = tqdm(hf_dataset, desc="Generating Triplets", 
                        unit="query", disable=is_iterable)

    try:
        for example in data_iterator:
            processed_count += 1
            query_text = example.get('query')
            passages_data = example.get('passages')

            # Basic validation
            if (not query_text or not isinstance(query_text, str) or
                not passages_data or not isinstance(passages_data, dict)):
                skipped_count += 1
                continue

            passage_texts = passages_data.get('passage_text', [])
            is_selected_flags = passages_data.get('is_selected', [])

            # Ensure both are lists and have the same length
            if (not isinstance(passage_texts, list) or
                not isinstance(is_selected_flags, list) or
                len(passage_texts) != len(is_selected_flags)):
                logger.debug(f"🔍 Skipping example: Invalid passage data.")
                skipped_count += 1
                continue

            # Find indices of positive and negative passages
            positive_indices = [i for i, selected in enumerate(is_selected_flags)
                              if selected == 1]
            negative_indices = [i for i, selected in enumerate(is_selected_flags)
                              if selected == 0]

            # --- Crucial Checks ---
            if not positive_indices or not negative_indices:
                skipped_count += 1
                continue # Cannot form a triplet

            # --- Select One Positive and One Negative ---
            chosen_pos_idx = random.choice(positive_indices)
            positive_text = passage_texts[chosen_pos_idx]

            chosen_neg_idx = random.choice(negative_indices)
            negative_text = passage_texts[chosen_neg_idx]

            # Check if selected texts are valid strings
            if not isinstance(positive_text, str) or not isinstance(negative_text, str):
                logger.debug(f"🔍 Skipping triplet: Invalid passage text type.")
                skipped_count += 1
                continue

            # --- Tokenize the selected texts ---
            query_ids = tokenize_text(query_text, vocab, max_query_len)
            pos_doc_ids = tokenize_text(positive_text, vocab, max_doc_len)
            neg_doc_ids = tokenize_text(negative_text, vocab, max_doc_len)

            # Optional: Skip if any tokenization resulted in empty lists
            if not query_ids or not pos_doc_ids or not neg_doc_ids:
                logger.debug(f"🔍 Skipping triplet: Empty tokenization result.")
                skipped_count +=1
                continue

            tokenized_triplets.append({
                'query': query_ids,
                'pos_doc': pos_doc_ids,
                'neg_doc': neg_doc_ids
            })

            # Check if max_triplets limit is reached
            if max_triplets is not None and len(tokenized_triplets) >= max_triplets:
                logger.info(f"✅ Reached max_triplets limit ({max_triplets}).")
                break
    except Exception as e:
        logger.error(f"❌ Error during triplet generation: {e}", exc_info=True)
    finally:
        if not is_iterable: data_iterator.close() # Close tqdm iterator

    logger.info(f"✅ Triplet generation complete. Processed: {processed_count}, "
               f"Generated: {len(tokenized_triplets)}, Skipped: {skipped_count}")
    if not tokenized_triplets:
        logger.warning("⚠️ No triplets were generated. Check data integrity.")
    return tokenized_triplets


# --- PyTorch Dataset Class ---
class TripletDataset(Dataset):
    '''PyTorch Dataset for (query, positive_doc, negative_doc) triplets.'''
    def __init__(self, tokenized_triplets: List[Dict[str, List[int]]]):
        self.tokenized_triplets = tokenized_triplets
        if not tokenized_triplets:
            logger.warning("⚠️ Initializing TripletDataset with empty list.")
        logger.info(f"📊 TripletDataset created with {len(self.tokenized_triplets)} triplets.")

    def __len__(self):
        return len(self.tokenized_triplets)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        return self.tokenized_triplets[idx]

# --- Collate Function ---
def collate_triplets(batch: List[Dict[str, List[int]]], 
                    padding_value: int = 0) -> Dict[str, torch.Tensor]:
    '''
    Collates a batch of triplets, padding sequences to the max length in batch.
    '''
    keys = ['query', 'pos_doc', 'neg_doc']
    # Filter out invalid items before calculating max_lens
    valid_batch = [
        item for item in batch 
        if isinstance(item, dict) and 
        all(k in item and isinstance(item[k], list) for k in keys)
    ]

    if not valid_batch:
        # Return empty tensors with appropriate keys
        return {f"{key}_ids": torch.empty((0, 0), dtype=torch.long) 
                for key in keys}

    try:
        # Calculate max lengths only from valid items
        max_lens = {
            key: max(len(item[key]) for item in valid_batch) 
            if valid_batch else 0 for key in keys
        }
    except ValueError as e:
        logger.error(f"❌ Error calculating max lengths: {e}")
        return {f"{key}_ids": torch.empty((0,0), dtype=torch.long) 
                for key in keys}

    padded_batch = {f"{key}_ids": [] for key in keys}

    # Iterate through batch, use max_lens from valid items
    for item in batch:
        if (isinstance(item, dict) and 
            all(k in item and isinstance(item[k], list) for k in keys)):
            # Process valid item
            for key in keys:
                ids = item[key]
                pad_len = max_lens[key] - len(ids)
                padded_ids = ids + [padding_value] * pad_len
                padded_batch[f"{key}_ids"].append(padded_ids)
        else:
            # Handle invalid item structure
            logger.debug(f"🔍 Invalid item format in batch. Appending padding.")
            for key in keys:
                padded_batch[f"{key}_ids"].append([padding_value] * max_lens[key])

    # Convert lists of lists to tensors
    output_tensors = {}
    try:
        for key in keys:
            list_of_ids = padded_batch[f"{key}_ids"]
            if list_of_ids: # Ensure list is not empty
                output_tensors[f"{key}_ids"] = torch.tensor(
                    list_of_ids, dtype=torch.long
                )
            else:
                output_tensors[f"{key}_ids"] = torch.empty(
                    (0, max_lens.get(key, 0)), dtype=torch.long
                )
    except Exception as e:
        logger.error(f"❌ Error converting batch lists to tensors: {e}")
        return {f"{key}_ids": torch.empty((0,0), dtype=torch.long) 
                for key in keys}

    return output_tensors


# --- Main execution block for testing ---
if __name__ == "__main__":
    logger.info("🧪 Running dataset.py script directly for testing...")

    # 1. Load configuration (needed for paths)
    config = load_config() # Assumes config.yaml is in project root
    if not config:
        logger.error("❌ Could not load config.yaml. Exiting.")
        sys.exit(1)

    paths_config = config.get('paths', {})
    vocab_path = paths_config.get('vocab_file')
    if not vocab_path or not os.path.exists(vocab_path):
        logger.error(f"❌ Vocabulary file path not found: {vocab_path}")
        sys.exit(1)

    # 2. Load Vocabulary
    try:
        vocab = Vocabulary.load_vocab(vocab_path)
        logger.info(f"📚 Vocabulary loaded successfully ({len(vocab)} words).")
        # For triplet generation from MS MARCO, we don't need sampling weights
        if vocab.sampling_weights is None:
            logger.warning("⚠️ Loaded vocab does not have sampling weights.")

    except Exception as e:
        logger.error(f"❌ Failed to load vocabulary: {e}", exc_info=True)
        sys.exit(1)

    # 3. Load a small portion of MS MARCO data for testing
    test_split = 'train' # Use train split for testing triplet generation
    hf_dataset_stream = load_msmarco_hf(split=test_split, streaming=True)

    if hf_dataset_stream:
        # Take first N items for testing
        num_test_items = 1000
        try:
            test_dataset_items = list(hf_dataset_stream.take(num_test_items))
            # Create a non-streaming dataset from these items
            from datasets import Dataset as HFBuildDataset
            test_dataset = HFBuildDataset.from_list(test_dataset_items)
            logger.info(f"📊 Created test dataset with {len(test_dataset)} examples.")

        except Exception as e:
            logger.error(f"❌ Failed to create test dataset: {e}", exc_info=True)
            sys.exit(1)

        # 4. Generate Triplets
        tower_config = config.get('two_tower', {})
        max_q_len = tower_config.get('max_query_len', 64)
        max_d_len = tower_config.get('max_doc_len', 256)

        # Generate a limited number of triplets for the test
        tokenized_triplets = generate_triplets_from_dataset(
            test_dataset, # Use the small, non-streaming test dataset
            vocab,
            max_q_len,
            max_d_len,
            max_triplets=500 # Generate max 500 triplets from 1000 examples
        )

        # 5. Test TripletDataset and DataLoader
        if tokenized_triplets:
            pytorch_dataset = TripletDataset(tokenized_triplets)
            logger.info(f"📊 Created TripletDataset with {len(pytorch_dataset)} items.")

            # Determine padding index
            padding_idx = vocab.unk_index # Use UNK index as padding

            # Test DataLoader with collate function
            try:
                test_dataloader = DataLoader(
                    pytorch_dataset,
                    batch_size=4, # Small batch size for testing
                    shuffle=False, # No need to shuffle for test
                    collate_fn=lambda batch: collate_triplets(
                        batch, padding_value=padding_idx
                    )
                )

                logger.info("🧪 Testing DataLoader iteration...")
                batch_count = 0
                for i, batch in enumerate(test_dataloader):
                    batch_count += 1
                    logger.info(f"  🧪 Batch {i+1}:")
                    for key, tensor in batch.items():
                        logger.info(f"    📊 {key}: shape={tensor.shape}")
                    if i == 0: # Print details only for the first batch
                        logger.info(f"    📊 First query_ids: {batch['query_ids'][0][:20]}...")
                    if i >= 2: # Stop after a few batches
                        break
                logger.info(f"✅ DataLoader test completed ({batch_count} batches).")

            except Exception as e:
                logger.error(f"❌ Error during DataLoader test: {e}", exc_info=True)

        else:
            logger.warning("⚠️ No tokenized triplets generated for testing.")
    else:
        logger.error("❌ Failed to load MS MARCO stream for testing.")

    logger.info("✅ Dataset.py test execution finished.")