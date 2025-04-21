#!/usr/bin/env python3
"""
Script to inspect the MS MARCO HuggingFace dataset in the style of project scripts.
Prints type, attributes, features, and sample entries for debugging data pipeline.
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset, Dataset as HFDataset

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    print(f"[Inspect Script] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)

# Import logger from utils
try:
    from utils import logger
except ImportError:
    import logging
    logger = logging.getLogger("InspectScript")
    logging.basicConfig(level=logging.INFO)
    logger.warning("⚠️ Could not import project logger; using basic logging.")


def main():
    try:
        logger.info("📥 Loading MS MARCO dataset (train split) in streaming mode...")
        dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)

        logger.info(f"🔍 Dataset type: {type(dataset)}")
        logger.info("📋 Available attributes and methods:")
        for attr in dir(dataset):
            if not attr.startswith("_"):
                attr_type = type(getattr(dataset, attr))
                logger.info(f"- {attr}: {attr_type}")

        logger.info(f"🌊 Is streaming dataset: {getattr(dataset, 'is_streaming', False)}")

        # Print features if available
        if hasattr(dataset, 'features'):
            logger.info("✨ Dataset features:")
            for name, feature in dataset.features.items():
                logger.info(f"- {name}: {feature}")
        else:
            logger.info("❓ (Features not available in streaming mode before iteration)")

        # Print a few example entries
        logger.info("🧪 Example entries (first 3):")
        for i, example in enumerate(dataset.take(3)):
            logger.info(f"\n📄 Example {i+1}:")
            for k, v in example.items():
                if isinstance(v, str) and len(v) > 100:
                    logger.info(f"  {k}: {v[:100]}...")
                else:
                    logger.info(f"  {k}: {v}")

        # Get a single sample and examine in detail
        logger.info("\n🔬 Detailed structure of a sample entry:")
        sample = next(iter(dataset.take(1)))
        for key, value in sample.items():
            logger.info(f"\n🔑 Key: {key}")
            logger.info(f"📊 Type: {type(value)}")
            if hasattr(value, "__len__"):
                logger.info(f"📏 Length: {len(value)}")
            if isinstance(value, list):
                if len(value) > 0:
                    logger.info(f"🧩 First element type: {type(value[0])}")
                    if len(value) > 2:
                        logger.info(f"👀 Preview: {value[:2]} ... (total: {len(value)} items)")
                    else:
                        logger.info(f"📜 All values: {value}")
            elif isinstance(value, str):
                if len(value) > 100:
                    logger.info(f"📝 Preview: {value[:100]}...")
                else:
                    logger.info(f"📝 Value: {value}")
            else:
                logger.info(f"📝 Value: {value}")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()