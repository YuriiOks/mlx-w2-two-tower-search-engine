# Week 2 Dev Plan: Two-Tower Search Engine (MS MARCO) 🗼🗼

**Team:** Perceptron Party (Yurii, Pyry, Dimitris, Dimitar)
**Goal:** Implement and train two types of Two-Tower models (RNN-based and Simple Average Pooling) using Triplet Loss on MS MARCO, leveraging pre-trained embeddings from Week 1.

*(Collaboration: Please sync frequently, especially between data prep and model implementation teams! Pair programming encouraged.)*


**Revised Dev Plan: Days 2 & 3**
---

**Context:** End-to-end pipeline is running. RNN model trains, saves, logs to W&B. Main issue: Loss flatlines at the margin value (currently 0.3), suggesting easy negatives or model limitations.

**Goal:** Diagnose and fix the flatlining loss, achieve meaningful training progress evidenced by improving validation metrics. Start preparing for efficient inference.

**Assignees:** Pyry & Yurii (Working together)

---
**Day 2: Diagnose Loss, Enhance Model, Implement Validation**
---

| Time Block  | Task Area                 | Specific Task                                          | Description & Details                                                                                                                                                                                          | Git Branch Suggestion                | Status      |
| :---------- | :------------------------ | :----------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------- | :---------- |
| **Morning** | **Branch & Setup** 🌿     | **Create New Branch from `main`**                        | Create a new branch (e.g., `feat/improve-training`) for today's work. Ensure `main` is up-to-date.                                                                                                           | `feat/improve-training`            | ✅ Done      |
| Morning     | **Validation Metrics** 📊   | **Implement Recall@k & MRR**                           | Create functions (likely in a new `src/two_tower/evaluation.py` or within `trainer.py`) to calculate Recall@k (e.g., k=1, 5, 10) and MRR. This requires comparing query embedding to *all* relevant doc embeddings for validation set. | `feat/improve-training`            | ⏳ To Do     |
| Morning     | **Validation Loop** 🔄      | **Add Validation Step to Trainer**                     | Modify `train_two_tower_model` in `trainer.py`: Add a loop after each training epoch to run evaluation on the validation split (load val data, run model in `eval()` mode, compute distances, use metrics functions). Log metrics to W&B. | `feat/improve-training`            | ⏳ To Do     |
| **Afternoon** | **Model Enhancement** 🚀  | **Unfreeze Embeddings & Use BiRNN**                  | Modify `config.yaml`: Set `embeddings.freeze: False` **AND** `two_tower.bidirectional: True`. Consider `two_tower.rnn_type: 'GRU'` or `'LSTM'`. **Crucial:** Plan to use a very low LR (e.g., `1e-5`) for the next run. | `feat/improve-training`            | ⏳ To Do     |
| Afternoon | **Training Tuning** ⚙️     | **Add Gradient Clipping & Scheduler (Optional)**       | In `trainer.py` (`train_epoch_two_tower`), add `torch.nn.utils.clip_grad_norm_` before `optimizer.step()`. *Optional:* Research and add a simple LR scheduler (e.g., `ReduceLROnPlateau` wrapping the optimizer). | `feat/improve-training`            | ⏳ To Do     |
| Afternoon | **Test Run & Analysis** 🔥| **Run Enhanced Training w/ Validation**                | Execute `scripts/train_two_tower.py` with the enhanced model config (BiRNN, unfrozen embeds), gradient clipping, **low LR (`--lr 1e-5`)**, and maybe 3-5 epochs.                                          | *(Run on `feat/improve-training`)* | ⏳ To Do     |
| End of Day  | **Sync & Review** 🧐      | **Analyze W&B: Loss & Validation Metrics**             | Carefully review W&B plots: Did batch loss drop below margin? Did Recall/MRR *improve* over epochs, even slightly? Discuss findings. Commit & push branch.                                                    | *(Review results)*                 | ⏳ To Do     |

---
**Day 3: Harder Negatives & Inference Prep**
---

| Time Block  | Task Area                 | Specific Task                                          | Description & Details                                                                                                                                                                                                                                   | Git Branch Suggestion             | Status      |
| :---------- | :------------------------ | :----------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------- | :---------- |
| **Morning** | **Hard Negatives (Try 1)** 😈 | **Implement In-Batch Negatives**                       | Modify `train_epoch_two_tower` and `calculate_distance`. For each query `q_i` in a batch, calculate its distance to its positive `p_i` *and* its distances to all other positives `p_j` (where `i != j`) in the batch (treating them as negatives). Adjust loss calculation. | `feat/in-batch-negatives`         | ⏳ To Do     |
| Morning     | **Test Run & Analysis** 🔥| **Run Training with In-Batch Negatives**               | Execute `scripts/train_two_tower.py` with the in-batch negative changes (keep enhanced model settings). Run a few epochs. Compare loss and validation metrics on W&B to previous runs. Does this provide a stronger signal?                                       | *(Run on `feat/in-batch-negatives`)* | ⏳ To Do     |
| **Afternoon** | **Inference Prep (Part 1)** 💾 | **Implement Corpus Embedding Script**                  | Create `scripts/embed_corpus.py`. This script should: load config, load trained `TwoTowerModel` weights (from a good checkpoint), load the full MS MARCO corpus (or passages), tokenize, run through the *document encoder*, and save all embeddings + IDs to a file. | `feat/corpus-embedding`           | ⏳ To Do     |
| Afternoon | **Inference Prep (Part 2)** 🔍 | **Basic Vector Indexing (FAISS/ChromaDB)**             | Install FAISS (`faiss-cpu` or `faiss-gpu`) or `chromadb`. Modify/extend `embed_corpus.py` or create a new script to load the saved document embeddings and build a simple ANN index (e.g., `IndexFlatL2` or `IndexFlatIP` for FAISS). Test basic search.          | `feat/vector-db-intro`            | ⏳ To Do     |
| End of Day  | **Wrap-up & Plan Next** 🎯 | **Review Progress, Merge Good Changes, Plan Week 3** | Review results of hard negative strategy and inference prep. Merge successful feature branches to `main` via PRs. Clean up code. Outline any remaining W2 tasks and plan initial steps for Week 3 (Transformers).                                     | Team (All)                        | `main` (via PRs)                  | ⏳ To Do     |
