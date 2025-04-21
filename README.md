<!-- Badges -->
<p align="center">
    <a href="https://github.com/YuriiOks/mlx-w2-two-tower-search-engine"><img src="https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github" alt="GitHub Repository"/></a>
    <img src="https://img.shields.io/github/contributors/YuriiOks/mlx-w2-two-tower-search-engine?style=for-the-badge" alt="Contributors"/>
    <img src="https://img.shields.io/github/issues/YuriiOks/mlx-w2-two-tower-search-engine?style=for-the-badge" alt="Issues"/>
    <img src="https://img.shields.io/github/license/YuriiOks/mlx-w2-two-tower-search-engine?style=for-the-badge" alt="License"/>
</p>


</p>

# MS MARCO Search Engine 🗼🗼 🔍 🧠

[![MLX Institute Logo](https://ml.institute/logo.png)](http://ml.institute)

A project by **Team Perceptron Party** 🎉 (Yurii, Pyry, Dimitris, Dimitar) for Week 2 of the MLX Institute Intensive Program.

## Project Overview 📋

This project builds a semantic search engine using a **Two-Tower Neural Network** 🏗️. The goal is to retrieve relevant documents from the MS MARCO v1.1 dataset based on user queries by learning dense vector representations for queries and documents.

The core components are:
1.  **Pre-trained Embeddings:** 📚 Leveraging word embeddings generated in Week 1 (Word2Vec - SkipGram trained on `text8`).
2.  **Two-Tower Architecture:** 🏢🏢 Uses separate RNN-based encoders (configurable, default: RNN/GRU) to learn embeddings for queries and documents.
3.  **Metric Learning:** 📏 Employs Triplet Margin Loss with Cosine Distance to train the towers, pushing query vectors closer to relevant document vectors and further from irrelevant ones.
4.  **Efficient Inference (Goal):** ⚡ Plan to pre-calculate document embeddings for fast retrieval using similarity search (e.g., via FAISS or ChromaDB).

## Key Features & Modules 🛠️

*   **Configuration:** ⚙️ Centralized parameters via `config.yaml` (model type, dimensions, LR, margin, paths, etc.).
*   **Data Pipeline (`src/two_tower/dataset.py`):** 🔄 Handles loading MS MARCO v1.1 (via `datasets` library), generating (Query, Positive Doc, Negative Doc) triplets based on `is_selected` flags, tokenization using Week 1 vocabulary, padding, and creating PyTorch `DataLoader`.
*   **Model Architecture (`src/two_tower/model.py`):** 🧩 Defines `TwoTowerModel` with `QueryEncoder` and `DocumentEncoder` (RNN/GRU/LSTM based, potentially shared), integrating pre-trained embeddings (with option to freeze/unfreeze).
*   **Training (`src/two_tower/trainer.py`):** 🏋️‍♀️ Implements the training loop using Triplet Loss, Adam optimizer, and W&B batch/epoch logging.
*   **Utilities (`utils/`):** 🔧 Shared functions for logging, device (CPU/MPS) setup, config loading, and artifact saving.
*   **Experiment Tracking (`wandb`):** 📊 Integrated for logging hyperparameters, metrics (batch/epoch loss), and saving artifacts (models, plots, config).
*   **Main Script (`scripts/train_two_tower.py`):** 🚀 Orchestrates loading, setup, training, and saving process.

## Current Status & Observations 📈📉

*   ✅ **End-to-End Pipeline:** The full pipeline from data loading to model training and artifact saving is functional.
*   ✅ **Bug Fixes:** Initial issues related to argument parsing, save paths, logging setup, and DataLoader pickling have been resolved.
*   ✅ **Training Runs:** Initial training experiments have been successfully executed on the full MS MARCO training set using an RNN encoder with frozen embeddings.
*   ⚠️ **Suspicious Loss Behavior:** Training loss (batch and epoch) drops rapidly in the first epoch but then **flatlines precisely at the configured margin value** (e.g., 0.2 or 0.3).
*   🤔 **Hypothesis:** This suggests the current model setup (simple RNN, frozen embeddings) combined with the default negative sampling (randomly chosen `is_selected=0` passages) makes it too easy for the model to satisfy the margin requirement. The model isn't receiving a strong enough signal to learn finer-grained distinctions after the initial separation.

## Directory Structure 📁

A detailed breakdown is available in `docs/STRUCTURE.MD`. Auto-generated documentation is available in `PROJECT_DOCUMENTATION.md/.html`.

## Setup 💻

1.  **Clone the Repository:** 📥
    ```bash
    git clone https://github.com/ocmoney/perceptron-party-search.git # Or your repo URL
    cd perceptron-party-search
    ```
2.  **Create & Activate Virtual Environment:** 🐍
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `\.venv\Scripts\activate`
    ```
3.  **Install Dependencies:** 📦 (Includes `datasets`, `torch`, `wandb`, `pyyaml`, etc.)
    ```bash
    pip install -r requirements.txt
    # May need: pip install pyarrow (for efficient HF dataset handling)
    ```
4.  **Weights & Biases Login:** 🔑
    ```bash
    wandb login
    ```
5.  **MS MARCO Data:** 🗃️ The first run of `scripts/train_two_tower.py` will automatically download and cache the MS MARCO v1.1 dataset via the `datasets` library. Ensure you have sufficient disk space in your home directory (`~/.cache/huggingface/datasets`).
6.  **Configure Word2Vec Paths:** ⚠️
    *   **CRITICAL:** Edit `config.yaml`.
    *   Verify `paths.vocab_file` points to the correct `*.json` vocabulary file from your Week 1 project.
    *   Verify `paths.pretrained_embeddings` points to the correct `.pth` model state file (e.g., the SkipGram one) from your Week 1 project.
    *   Verify `paths.two_tower_model_save_dir` points to `models/two_tower`.

## Usage 🚦

1.  **Configuration:** ⚙️ Review and adjust parameters in `config.yaml` (especially `embeddings.freeze`, `two_tower.rnn_type`, `two_tower.bidirectional`, `two_tower_training.learning_rate`, `two_tower_training.margin`).
2.  **Run Training:** 🏃‍♂️ Execute the main training script from the project root directory:
    ```bash
    # Example: Train for 3 epochs with specific LR, overriding config defaults
    python scripts/train_two_tower.py --epochs 3 --lr 1e-5

    # Example: Use all config defaults (check config first!)
    # python scripts/train_two_tower.py
    ```
    *   Training progress (`tqdm` bars) shown in console.
    *   Metrics logged to Weights & Biases (link provided in console).
    *   Model artifacts saved locally under `models/two_tower/<W&B_RUN_NAME>/`.

Use `python scripts/train_two_tower.py --help` for all command-line options.

## Next Steps & Future Work 🔮

*   ➡️ **Investigate Flatlining Loss:**
    *   **Experiment with Model Configuration:**
        *   **Unfreeze Embeddings:** Set `embeddings.freeze: False` in `config.yaml` and use a very low learning rate (e.g., `--lr 1e-5` or `5e-6`). **(High Priority)**
        *   **Use Bi-Directional RNNs:** Set `two_tower.bidirectional: True` and `two_tower.rnn_type: 'GRU'` or `'LSTM'` in `config.yaml`. **(High Priority)**
        *   Add Layer Normalization / Dropout within RNNs if needed later.
*   ➡️ **Implement Harder Negative Sampling:**
    *   **In-Batch Negatives:** Modify training loop/loss calculation to use other positive documents within the same batch as negatives.
    *   **Offline Hard Negatives:** Explore generating negatives based on BM25 scores (if retrievable) or ANN search using a previously trained model checkpoint.
*   ➡️ **Implement Validation:**
    *   Add evaluation loop using the MS MARCO validation split.
    *   Calculate and log standard IR metrics like **Recall@k** and **MRR@k** to W&B to track actual retrieval performance.
*   **Inference Pipeline:** ⚡
    *   Develop script to pre-compute and save embeddings for the entire document corpus.
    *   Integrate **FAISS** or **ChromaDB** for efficient ANN search during inference.
    *   Build the query encoding and search logic (`scripts/evaluate_two_tower.py` or separate inference script).
*   **Serving:** 🌐 Connect the trained model and inference logic to the `app/app.py` Streamlit app or a FastAPI service.
*   **Containerization:** 🐳 Finalize the `Dockerfile` for deployment.