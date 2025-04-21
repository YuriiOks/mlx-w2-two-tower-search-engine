<!-- Badges -->
<p align="center">
    <a href="https://github.com/YuriiOks/mlx-w2-two-tower-search-engine"><img src="https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github" alt="GitHub Repository"/></a>
    <img src="https://img.shields.io/github/contributors/YuriiOks/mlx-w2-two-tower-search-engine?style=for-the-badge" alt="Contributors"/>
    <img src="https://img.shields.io/github/issues/YuriiOks/mlx-w2-two-tower-search-engine?style=for-the-badge" alt="Issues"/>
    <img src="https://img.shields.io/github/license/YuriiOks/mlx-w2-two-tower-search-engine?style=for-the-badge" alt="License"/>
</p>

# MS MARCO Search Engine 🗼🗼 🔍 🧠

[![MLX Institute Logo](https://ml.institute/logo.png)](http://ml.institute)

A project by **Team Perceptron Party** 🎉 (Yurii, Pyry, Dimitris, Dimitar) for Week 2 of the MLX Institute Intensive Program.

## Project Overview 📋

This project builds a semantic search engine using a **Two-Tower Neural Network** 🏗️. The goal is to retrieve relevant documents from the MS MARCO dataset based on user queries.

The core components are:
1.  **Pre-trained Embeddings:** 📚 Leveraging word embeddings generated in Week 1 (Word2Vec - CBOW/SkipGram trained on `text8`).
2.  **Two-Tower Architecture:** 🏢🏢 Separate RNN-based encoders (e.g., GRU, LSTM) learn dense vector representations for queries and documents.
3.  **Metric Learning:** 📏 Uses Triplet Loss to train the towers, pushing query vectors closer to relevant document vectors and further from irrelevant ones in the embedding space.
4.  **Efficient Inference:** ⚡ Pre-calculates document embeddings for fast retrieval using similarity search, potentially leveraging a vector database like ChromaDB.

## Key Features & Modules 🛠️

*   **Configuration:** ⚙️ Centralized parameters via `config.yaml`.
*   **Data Handling (`src/two_tower/dataset.py`):** 🔄 Placeholder for MS MARCO loading, tokenization (using W1 vocab), triplet generation, padding, and PyTorch Dataset/DataLoader creation.
*   **Model Architecture (`src/two_tower/model.py`):** 🧩 Defines `TwoTowerModel` with separate or shared `QueryEncoder` and `DocumentEncoder` using configurable RNN types, integrating pre-trained embeddings.
*   **Training (`src/two_tower/trainer.py`):** 🏋️‍♀️ Implements the training loop using Triplet Loss and specified distance metric (e.g., cosine similarity).
*   **Utilities (`utils/`):** 🔧 Shared functions for logging, device (CPU/MPS) setup, config loading, and artifact saving.
*   **Experiment Tracking (`wandb`):** 📊 Integrated for logging hyperparameters, metrics (loss), and saving artifacts (models, vocab, results).
*   **Main Script (`scripts/train_two_tower.py`):** 🚀 Orchestrates the loading, setup, training, and saving process.

## Directory Structure 📁

A detailed breakdown is available in `docs/STRUCTURE.md`.

## Setup 💻

1.  **Clone the Repository:** 📥
```bash
git clone https://github.com/ocmoney/perceptron-party-search.git # Replace with your actual repo URL if different
cd perceptron-party-search
```
2.  **Initialize Git (if not cloned):** 🌱
```bash
git init && git add . && git commit -m "Initial project structure"
```
3.  **Create & Activate Virtual Environment:** 🐍
```bash
# Using venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `\.venv\Scripts\activate`

# Or using Conda
# conda create -n pparty python=3.11 # Or your preferred version
# conda activate pparty
```
4.  **Install Dependencies:** 📦
```bash
pip install -r requirements.txt
```
5.  **Weights & Biases Login:** 🔑
```bash
wandb login
```
(Follow prompts to paste your API key from wandb.ai/authorize)

6.  **Download MS MARCO Data:** 🗃️
        *   Obtain the MS MARCO v1.1 dataset (e.g., via Hugging Face `datasets` library or direct download).
        *   Place the necessary files (like training triples) into the `data/msmarco/` directory.
        *   Update the paths in `config.yaml` (`paths.train_triples`, `paths.val_triples`) if needed.

7.  **Configure Word2Vec Paths:** ⚠️
        *   **CRITICAL:** Edit `config.yaml`.
        *   Update `paths.vocab_file` to the correct path of the `*.json` vocabulary file from your Week 1 project.
        *   Update `paths.pretrained_embeddings` to the correct path of the `.pth` model state file containing the embeddings you want to use (e.g., the SkipGram one) from your Week 1 project.

## Usage 🚦

1.  **Configuration:** ⚙️ Review and adjust parameters in `config.yaml` (e.g., `two_tower` settings like `rnn_type`, `hidden_dim`; `training` settings like `epochs`, `lr`, `margin`).
2.  **Run Training:** 🏃‍♂️ Execute the main training script from the project root directory:
```bash
python scripts/train_two_tower.py
```
        *   Training progress will be shown in the console (including `tqdm` bars).
        *   Metrics and configuration will be logged to Weights & Biases. A link to the run will be printed.
        *   The trained model and artifacts will be saved locally in a run-specific subdirectory under `models/two_tower/`.

3.  **Override Config (Optional):** 🔄 Use command-line arguments to override `config.yaml` settings for specific runs:
```bash
# Example: Train for more epochs with a smaller learning rate
python scripts/train_two_tower.py --epochs 10 --lr 0.0001

# Example: Disable W&B logging for a quick test
python scripts/train_two_tower.py --epochs 1 --no-wandb
```
        Use `python scripts/train_two_tower.py --help` for all options.

4.  **Evaluation:** 📈 Run the evaluation script (once implemented):
```bash
python scripts/evaluate_two_tower.py --run-dir models/two_tower/TwoTower_RNN_... # Path to saved run
        ```

## Future Work & Considerations 🔮

*   **Implement Data Loading:** 📥 Replace placeholder functions in `src/two_tower/dataset.py` with actual MS MARCO loading, tokenization, and triplet generation logic.
*   **Refine Model/Trainer:** 🔧 Implement RNN variants (LSTM, BiRNN), Layer Normalization, Dropout as needed based on performance. Consider packed sequences for variable lengths.
*   **Evaluation Metrics:** 📊 Implement Recall@k, MRR@k, or other relevant information retrieval metrics in `scripts/evaluate_two_tower.py`.
*   **Inference Pipeline:** ⚡
        *   Develop a script/process to pre-compute and store embeddings for all documents in the MS MARCO corpus.
        *   **Vector Database:** 🔍 Integrate a vector database like **ChromaDB** or FAISS to store document embeddings for efficient similarity search during inference.
        *   Build the query encoding and search logic.
*   **FastAPI Integration:** 🌐 Connect the trained model and inference logic to the `app/` service.
*   **Containerization:** 🐳 Finalize the `Dockerfile`.