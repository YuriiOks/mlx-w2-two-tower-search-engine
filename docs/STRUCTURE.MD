# MS MARCO Search Engine - Project Structure

This document outlines the detailed directory structure and the purpose of all key files for the Week 2 Two-Tower Search Engine project by **Team Perceptron Party**.

## Directory Tree 🌳

```
📁 mlx-w2-two-tower-search-engine/
├── 📁 app/
├── ├── 📄 __init__.py
├── └── 📄 app.py
├── 📁 data/
├── └── 📁 msmarco/
├── 📁 data_triplets/
├── ├── 📄 test_triplets.csv
├── ├── 📄 train_triplets.csv
├── └── 📄 val_triplets.csv
├── 📁 docs/
├── ├── 📄 DEV_PLAN_W2.md
├── └── 📄 STRUCTURE.MD
├── 📁 logs/
├── 📁 models/
├── ├── 📁 two_tower/
├── ├── ├── 📁 TwoTower_E3_LR1e-05_BS128/
├── ├── ├── ├── 📄 training_loss.png
├── ├── ├── ├── 📄 training_losses.json
├── ├── ├── └── 📄 two_tower_final.pth
├── ├── └── 📁 TwoTower_E5_LR0.0001_BS128/
├── ├── └── ├── 📄 training_loss.png
├── ├── └── ├── 📄 training_losses.json
├── ├── └── └── 📄 two_tower_final.pth
├── └── 📁 word2vec/
├── └── ├── 📁 CBOW_D128_W5_NWAll_MF5_E15_LR0.001_BS512/
├── └── ├── ├── 📄 cbow_training_loss.png
├── └── ├── ├── 📄 cbow_training_losses.json
├── └── ├── └── 📄 model_state.pth
├── └── ├── 📁 SkipGram_D128_W5_NWAll_MF5_E3_LR0.001_BS512/
├── └── ├── ├── 📄 model_state.pth
├── └── ├── ├── 📄 training_loss.png
├── └── ├── └── 📄 training_losses.json
├── └── └── 📄 text8_vocab.json
├── 📁 notebooks/
├── 📁 scripts/
├── ├── 📄 evaluate_two_tower.py
├── ├── 📄 generate_project_doc.py
├── ├── 📄 inspect_hf_dataset.py
├── ├── 📄 train_two_tower.py
├── └── 📄 train_word2vec.py
├── 📁 src/
├── ├── 📁 two_tower/
├── ├── ├── 📄 __init__.py
├── ├── ├── 📄 dataset.py
├── ├── ├── 📄 model.py
├── ├── └── 📄 trainer.py
├── ├── 📁 word2vec/
├── ├── ├── 📄 __init__.py
├── ├── ├── 📄 dataset.py
├── ├── ├── 📄 model.py
├── ├── ├── 📄 trainer.py
├── ├── └── 📄 vocabulary.py
├── └── 📄 __init__.py
├── 📁 utils/
├── ├── 📄 __init__.py
├── ├── 📄 device_setup.py
├── ├── 📄 logging.py
├── └── 📄 run_utils.py
├── 📄 .flake8
├── 📄 .gitignore
├── 📄 Dockerfile
├── 📄 LICENSE
├── 📄 README.md
├── 📄 config.yaml
├── 📄 marco_entry.txt
├── 📄 ms_marco_entry.txt
├── 📄 requirements.txt
└── 📄 setup_project.py
```

## Detailed Directory & File Descriptions 📜

*   **Root Directory (`mlx-w2-two-tower-search-engine/`)**: Contains configuration, main scripts, documentation, and subdirectories for source code, data, models, etc.
    *   📄 **`.flake8`**: Configuration file for the Flake8 code style checker.
    *   📄 **`.gitignore`**: Specifies intentionally untracked files/directories that Git should ignore (e.g., virtual environments `.venv`, log files `logs/`, model artifacts `models/`, local W&B cache `wandb/`, Python cache `__pycache__/`).
    *   📄 **`config.yaml`** ⚙️: Central configuration file. Defines all crucial paths (data, vocab, embeddings, model saves), hyperparameters for Word2Vec (Week 1) and Two-Tower models (embeddings, architecture, training), and logging settings. Read by `utils/run_utils.py`.
    *   📄 **`Dockerfile`**: Instructions for building a Docker container image for the application/service, enabling deployment.
    *   📄 **`LICENSE`**: Contains the project's license information (e.g., MIT License).
    *   📄 **`marco_entry.txt`**, 📄 **`ms_marco_entry.txt`**: Appear to be text files containing sample data entries from the MS MARCO dataset, likely used for reference or manual inspection during development.
    *   📄 **`PROJECT_DOCUMENTATION.md`**, 📄 **`PROJECT_DOCUMENTATION.html`**: Auto-generated comprehensive documentation files created by `scripts/generate_project_doc.py`.
    *   📄 **`README.md`** 👋: The main entry point for understanding the project. Provides a high-level overview, setup instructions, usage examples, current status, and future work.
    *   📄 **`requirements.txt`** 📦: Lists all required Python packages and their versions needed to run the project. Used for setting up the environment (`pip install -r requirements.txt`). Includes libraries like `torch`, `datasets`, `wandb`, `PyYAML`, `pandas`, `numpy`, `tqdm`, etc.
    *   📄 **`setup_project.py`**: A utility script used initially to create the basic directory structure and template files for the project.

*   📁 **`app/`**: Contains code for the user-facing application or API service (likely using Streamlit or FastAPI later).
    *   📄 **`__init__.py`**: Marks the `app` directory as a Python package.
    *   📄 **`app.py`**: The main script for the application (currently contains a dummy Streamlit app). Will eventually integrate the trained model for inference.

*   📁 **`data/`**: Directory intended for storing input data.
    *   📁 **`msmarco/`** 📊: Designated location for storing the raw MS MARCO v1.1 dataset files downloaded (or cached) by the Hugging Face `datasets` library.

*   📁 **`data_triplets/`**: *New directory observed.* Appears to be intended for storing pre-generated (Query, Positive Passage, Negative Passage) triplets, potentially as CSV files. This might be used if triplet generation is done offline or if specific pre-defined splits are used.
    *   📄 **`test_triplets.csv`**, 📄 **`train_triplets.csv`**, 📄 **`val_triplets.csv`**: CSV files likely containing pre-generated query/passage IDs or text for training, validation, and testing using triplets. *(Check if `dataset.py` uses these or generates triplets dynamically)*.

*   📁 **`docs/`** 📄: Contains project documentation files.
    *   📄 **`DEV_PLAN_W2.md`**: The development plan outlining tasks, assignments, and schedule for Week 2.
    *   📄 **`STRUCTURE.MD`**: This file, describing the project structure and file purposes.

*   📁 **`logs/`** 📝: Stores runtime log files generated by the application (configured in `utils/logging.py`). (Gitignored).

*   📁 **`models/`** 🧠: Root directory for storing saved model artifacts. (Gitignored).
    *   📁 **`two_tower/`**: Contains subdirectories specific to trained Two-Tower model runs.
        *   📁 **`<RUN_NAME>/`** (e.g., `TwoTower_E3_LR1e-05_BS128/`): A subdirectory created for each training run, typically named based on hyperparameters or W&B run name.
            *   📄 **`two_tower_final.pth`**: The saved state dictionary (weights) of the trained PyTorch Two-Tower model for that run.
            *   📄 **`training_losses.json`**: JSON file storing the list of average training losses per epoch for that run.
            *   📄 **`training_loss.png`**: Plot visualizing the training loss per epoch saved as an image.
    *   📁 **`word2vec/`**: Contains artifacts from the Week 1 Word2Vec training. These are *inputs* for the Week 2 project.
        *   📁 **`<RUN_NAME>/`** (e.g., `SkipGram_.../`): Subdirectory for a specific Word2Vec training run.
            *   📄 **`model_state.pth`**: Saved state dictionary of the *pre-trained Word2Vec embeddings* used by the Two-Tower model.
            *   📄 `training_loss.png`, `training_losses.json`: Loss artifacts from the Word2Vec run.
        *   📄 **`text8_vocab.json`**: The vocabulary file (mapping words to indices) generated during Word2Vec training, used for tokenizing MS MARCO data.

*   📁 **`notebooks/`** 📓: Location for Jupyter notebooks used for exploratory data analysis (EDA), model evaluation, visualization, or prototyping.

*   📁 **`scripts/`** ▶️: Contains the main executable Python scripts for performing key project tasks.
    *   📄 **`evaluate_two_tower.py`**: Script intended for evaluating the trained Two-Tower model using metrics like Recall@k and MRR. (Currently a placeholder).
    *   📄 **`generate_project_doc.py`**: Utility script to automatically generate `PROJECT_DOCUMENTATION.md` and `.html` files.
    *   📄 **`inspect_hf_dataset.py`**: Utility script for debugging and understanding the structure of the MS MARCO dataset loaded via Hugging Face `datasets`.
    *   📄 **`train_two_tower.py`**: The primary script to orchestrate the entire Two-Tower model training process: loads config, loads data/vocab/embeddings, initializes model/optimizer, calls the trainer, saves artifacts, logs to W&B.
    *   📄 **`train_word2vec.py`**: The script from Week 1 used to train the Word2Vec model (CBOW or SkipGram) that produces the pre-trained embeddings and vocabulary used in Week 2.

*   📁 **`src/`** 🐍: Contains the core source code organized into Python modules.
    *   📄 **`__init__.py`**: Makes `src` directory a Python package.
    *   📁 **`two_tower/`** ✨: Modules implementing the Two-Tower search model logic.
        *   📄 **`__init__.py`**: Marks `two_tower` as a sub-package.
        *   📄 **`dataset.py`**: Contains functions and classes for data handling: `load_msmarco_hf` (loads data), `generate_triplets_from_dataset` (creates Q,P,N triplets), `tokenize_text`, `TripletDataset` (PyTorch Dataset), and `collate_triplets` (batch padding).
        *   📄 **`model.py`**: Defines the PyTorch neural network architecture: `QueryEncoder` (RNN-based), `DocumentEncoder` (RNN-based, possibly shared), and the main `TwoTowerModel` class that integrates the embedding layer and the two encoders.
        *   📄 **`trainer.py`**: Implements the model training logic: `train_epoch_two_tower` (performs one epoch of training with Triplet Loss) and `train_two_tower_model` (orchestrates the overall training loop over multiple epochs, handles saving, calls epoch trainer).
    *   📁 **`word2vec/`**: Reused modules from Week 1 related to Word2Vec implementation (Vocabulary, Dataset generation, Model definition, Trainer).
        *   📄 `__init__.py`, `dataset.py`, `model.py`, `trainer.py`, `vocabulary.py`
    *   📄 **`__init__.py`**: *Duplicate entry in tree? Should likely only be one at `src/` level.*

*   📁 **`utils/`** 🛠️: Contains shared utility modules used across the project.
    *   📄 **`__init__.py`**: Marks `utils` as a package and potentially exposes key utility functions.
    *   📄 **`device_setup.py`**: Function (`get_device`) to detect and select the appropriate PyTorch compute device (CPU, MPS, CUDA).
    *   📄 **`logging.py`**: Configures the project-wide logger (`Perceptron Party`) with specified format, level, and handlers (console, rotating file).
    *   📄 **`run_utils.py`**: Helper functions for common tasks like loading the `config.yaml` file, saving/plotting training losses, and formatting numbers.

*   📁 **`wandb/`** ☁️: Directory created by Weights & Biases to store local run data, logs, and cache before syncing to the cloud. (Gitignored).

