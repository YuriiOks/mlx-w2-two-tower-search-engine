# MS MARCO Search Engine - Project Structure

This document outlines the directory structure and the purpose of key files for the Week 2 Two-Tower Search Engine project by **Team Perceptron Party**.

```
.
├── config.yaml              # ⚙️ Central configuration (paths, hyperparameters for W2)
├── data/                    # 📊 Input data
│   └── msmarco/             # MS MARCO specific data (needs population)
├── docs/                    # 📄 Project documentation
│   ├── STRUCTURE.md         # This file!
│   └── DEV_PLAN_W2.md       # Week 2 Development Plan & Tasks
├── logs/                    # 📝 Runtime logs (Gitignored)
├── models/                  # 🧠 Saved model artifacts (Gitignored)
│   └── two_tower/           # Run-specific subdirs for the Two-Tower model
│       └── TwoTower_RNN_.../ # Example run directory
│           ├── two_tower_final.pth
│           ├── training_losses.json
│           └── training_loss.png
├── notebooks/               # 📓 Jupyter notebooks for exploration & evaluation
│   └── W2_MSMARCO_EDA.ipynb # Placeholder for MS MARCO EDA
│   └── W2_Evaluation.ipynb  # Placeholder for evaluating retrieval
├── requirements.txt         # 📦 Python package dependencies
├── scripts/                 # ▶️ Runnable scripts
│   ├── train_two_tower.py   # Script to orchestrate Two-Tower training
│   └── evaluate_two_tower.py# Placeholder for evaluation script
│   └── (inference.py)       # Optional: Script for running inference later
├── setup_project.py         # The script used to generate this structure
├── src/                     # 🐍 Core Python source code modules
│   ├── __init__.py
│   └── two_tower/           # ✨ Two-Tower implementation modules
│       ├── __init__.py
│       ├── dataset.py       # MS MARCO data loading, triplet generation, Dataset/Collation
│       ├── model.py         # Two-Tower model definition (Embeddings, RNN Towers)
│       └── trainer.py       # Training loop logic with Triplet Loss
├── utils/                   # 🛠️ Shared utility modules
│   ├── __init__.py
│   ├── device_setup.py      # CPU/MPS/CUDA device selection
│   ├── logging.py           # Logging setup
│   └── run_utils.py         # Config loading, artifact saving helpers
├── .gitignore               # Files/directories for Git to ignore
├── README.md                # 👋 High-level project overview
└── wandb/                   # ☁️ Local W&B logs/cache (In .gitignore)
    └── ...
```

**Directory & Key File Purposes:**

*   **`config.yaml`** ⚙️: Central configuration for Week 2, including paths to MS MARCO data, Word2Vec artifacts, Two-Tower hyperparameters (RNN type, hidden dims), and training settings (Triplet Loss margin, LR, epochs).
*   **`data/msmarco/`** 📊: Designated location for storing the downloaded MS MARCO dataset files (v1.1 recommended). Needs to be populated.
*   **`docs/`** 📄: Project documentation specific to Week 2 (Structure, Dev Plan).
*   **`logs/`** 📝: Runtime logs for this project. Should be gitignored.
*   **`models/two_tower/`** 🧠: Will store run-specific subdirectories containing the trained Two-Tower model state (`.pth`), loss history, etc. Should be gitignored.
*   **`notebooks/`** 📓: For EDA specifically on the MS MARCO dataset and for evaluating the trained retrieval model.
*   **`scripts/`** ▶️: Contains the main runnable scripts for training (`train_two_tower.py`) and potentially evaluation/inference.
*   **`src/two_tower/`** ✨: Core implementation modules for the Two-Tower architecture, data handling (MS MARCO specifics, triplet generation), and the Triplet Loss training loop.
*   **`utils/`** 🛠️: Shared utilities (logging, device setup, config loading, artifact saving) reused from Week 1 setup.
*   **`README.md`** 👋: Updated overview for the Week 2 project, setup, and usage instructions.
*   **`requirements.txt`** 📦: Dependencies including PyTorch, PyYAML, W&B, and potentially Hugging Face `datasets`.
*   **`.gitignore`**: Essential for ignoring `logs/`, `models/`, `wandb/`, `.venv/`, `__pycache__/`, etc.
