# EvoTriKT

**Three-View Alignment with Language Model Evidence for Knowledge Tracing**

EvoTriKT combines interaction history, evolving concept states, and language-model
semantics to predict the correctness of a student's next response.

## Repository structure

```text
EvoTriKT/
├── models/
│   ├── EvoTriKT.py           # Model and its component layers
│   └── __init__.py
├── data_provider/
│   ├── paper_aligned_kt.py   # Shared reader for the four datasets
│   └── __init__.py
├── scripts/
│   └── train_config.json    # Training parameters
├── train.py                 # Training, validation, and final testing
├── requirements.txt
└── README.md
```

The model implementation is in `models/EvoTriKT.py`. Its sequence encoder,
concept-state updater, semantic encoder, alignment, routing, fusion, and prediction
layers are defined together in this file.

## Installation

Use Python 3.11. Install a PyTorch 2.5 build suitable for your CPU or CUDA device,
then install the remaining dependencies:

```bash
git clone https://anonymous.4open.science/r/EvoTriKT-A593.git
cd EvoTriKT
python -m pip install -r requirements.txt
```

## Data preparation

Supported dataset keys are `assist2009`, `nips34`, `xes3g5m`, and `ednet`.
Place your processed data and matching semantic vectors as follows:

```text
dataset/<dataset>/processed/
  <dataset>_paper_sequences_<tag>.jsonl
  <dataset>_paper_id_maps_<tag>.json
  <dataset>_paper_summary_<tag>.json
  graph_adjacency_tensors_<tag>.pt
semantic/<dataset>/pair_embeddings.pt
```

Sequence records contain `user_id`, `sequence_length`, `question_ids`,
`concept_set_ids`, `responses`, `timestamps`, `time_gap_seconds`, `duration_ms`,
`fixed_split`, and `fold`. The reader uses the stored student partitions and fold
labels. Semantic caches contain `pair_keys` and `pair_embeddings` with the same
question/concept mapping. The language features use frozen
`deepseek-ai/deepseek-llm-7b-chat` embeddings.

Datasets, semantic caches, and model weights are supplied separately.

##培训

Set training parameters in `scripts/train_config.json` and run:

```bash
python train.py --config scripts/train_config.json
```

Command-line arguments override the configuration file. For example:

```bash
python train.py --data_root dataset/nips34 --prompt_embedding_dir semantic/nips34 --save_dir Results/nips34/fold_0 --fold_idx 0
```

The dataset directory must use its supported key. `--processed_tag` must match
the processed filenames. To evaluate all five stored folds, run with
`--fold_idx 0` through `--fold_idx 4`, using a separate output directory for each.
The example configuration is editable; select the parameters matching your experiment.

Validation runs after each epoch. Validation AUC selects `best_model.pth`, and
the selected checkpoint is evaluated on the test set. The output directory contains:

```text
best_model.pth          # Validation-selected weights
metrics_so_far.json    # Training and validation progress
metrics.json           # Final AUC, ACC, F1, RMSE, NLL, and diagnostics
```

The configuration uses `patience=1`. Use `--device cpu` explicitly for CPU execution.
ACC and positive-class F1 use threshold 0.5. See `python train.py --help` for the
available parameters.
