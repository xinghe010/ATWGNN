# ATWGNNS

Reference implementation for the paper

> **ATWGNNS: Graph Simplified Representation and Learning of Logical Formula for Premise Selection**
> Xingxing He, Zhongxu Zhao, Yongqi Lan, Yingfang Li, Li Zou, Luis Martínez, Tianrui Li
> *Artificial Intelligence*.

ATWGNNS is an attention-based term-walk graph neural network with a simplified graph representation (S-DAGs) of first-order logic formulas, designed for premise selection in automated theorem proving over large theories.

The repository provides:

- **A simplified graph representation** that removes redundant repeated quantifier nodes while preserving the task-relevant structure of the original DAG.
- **A formula / graph similarity metric** that combines first-order unification with a restricted Weisfeiler-Lehman subtree kernel, giving the simplified graphs an interpretable foundation: higher logical similarity ⇒ higher graph similarity.
- **The ATWGNNS model**: a term-walk message-passing GNN with attention, trained as a binary classifier over (premise, conjecture) pairs.

## Repository structure

```
.
├── A-TWGNN-MPTP/              # MPTP experiment (formulas in general first-order form)
│   ├── dataset/               # raw json splits, statements file, node_dict.pkl
│   │   ├── node_dict.pkl
│   │   ├── statements
│   │   ├── train/raw/train.json
│   │   ├── valid/raw/valid.json
│   │   └── test/raw/test.json
│   ├── dataset.py             # PyG dataset + (premise, conjecture) pair construction
│   ├── formula_parser.py      # Lark-based FOF parser
│   ├── graph.py               # simplified DAG (S-DAGs) builder
│   ├── graph_DAGs.py          # standard DAG builder (baseline representation)
│   ├── model.py               # ATWGNNS modules and premise selection model
│   ├── trainer.py             # train / valid / test loops
│   ├── eval.py                # entry point (hyper-parameters + main loop)
│   └── utils.py               # logging, IO, plotting helpers
├── A-TWGNN-CNF/code/          # CNF experiment (formulas in conjunctive normal form)
│   └── ...                    # mirrors the MPTP layout
├── requirements.txt
└── README.md
```

## Environment

- Python 3.8 – 3.10
- A CUDA-capable GPU is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

`torch-scatter` and `torch-geometric` wheels depend on the exact PyTorch + CUDA combination. If the plain `pip install` fails, follow the official instructions:

- https://pytorch.org/get-started/locally/
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Datasets

The experiments use two derived versions of the **MPTP2078** question library (1469 conjectures, 24087 premises):

Both sub-projects ship the train / valid / test JSON splits (55 251 / 6 901 / 6 902 samples, an 80 / 10 / 10 random split totalling 69 054 pairs), the `statements` lookup table from formula name to TPTP text, and `node_dict.pkl` (793 token vocabulary, with all variables uniformly mapped to a single `Var` token).

A row in `train.json` looks like:

```json
["t14_funct_1", "t71_enumset1", 1]
```

i.e. `(conjecture_name, premise_name, label)` where `1` means the premise is useful for proving the conjecture (`0` otherwise). Useless premises were sampled by ranking the conjecture's unused premises with k-NN over symbol/subterm features and picking the top-ranked ones, so positive and negative samples are balanced.

## Quick start

### Train and evaluate on MPTP

```bash
cd A-TWGNN-MPTP
python eval.py \
    --root_dir ./dataset \
    --model_save ./model_save \
    --node_out_channels 512 \
    --layers 2 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --device cuda:0
```

### Train and evaluate on CNF

```bash
cd A-TWGNN-CNF/code
python eval.py \
    --root_dir ./data \
    --model_save ./model_save \
    --node_out_channels 128 \
    --layers 1 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --device cuda:0
```

### Command-line arguments

| Argument | MPTP default | CNF default | Description |
|----------|--------------|-------------|-------------|
| `--root_dir` | `./dataset` | `./data` | dataset root directory |
| `--model_save` | `./model_save` | `./model_save` | checkpoint / log output directory |
| `--node_out_channels` | 512 | 128 | node embedding dimension `d_{h_v}` |
| `--layers` | 2 | 1 | number of message-passing iterations `K` |
| `--epochs` | 100 | 100 | training epochs |
| `--batch_size` | 32 | 32 | mini-batch size |
| `--lr` | 0.001 | 0.001 | initial learning rate (decayed via `ReduceLROnPlateau`) |
| `--weight_decay` | 1e-4 | 1e-4 | L2 regularisation coefficient |
| `--device` | `cpu` | `cpu` | `cpu` or `cuda:N` |

### Outputs

The first run builds PyTorch-Geometric `processed/*.pt` caches under each split (this is the most time-consuming step). Subsequent runs reuse the cache.

Each run writes to `${model_save}/`:

- `record.log` — per-epoch loss / accuracy / F1 / recall / precision
- `best.pt` — best checkpoint (lowest validation loss)
- `history.pkl` — full metric history
- `figure` — learning-curve plot

## Hyper-parameter search space

The grid explored in the paper:

| Parameter | Searched values |
|-----------|------------------|
| Node vector dimension `d_{h_v}` | 128 / 256 / 512 |
| Message-passing iterations `K` | 1 / 2 / 3 |
| Training epochs | 100 / 150 / 200 |
| Batch size | 16 / 32 / 64 |
| Initial learning rate | 0.01 (decayed via `ReduceLROnPlateau`, min 1e-5) |
| Optimizer | Adam (default β's) |

The 793-dim one-hot input represents the token vocabulary; an embedding lookup `F_V` maps it to `d_{h_v}` before message passing. Three MLPs (`F_u`, `F_m`, `F_l`) aggregate term-walk neighbours from the upper / middle / lower position of each triple `(parent, node, child)`; attention weights are produced by softmax over the aggregated messages and combined by `F_sum` with a residual connection. Graph-level vectors are obtained by mean pooling and fed into a two-layer MLP classifier with batch normalisation.

## Main results

Premise classification on the held-out test split (higher is better):

**MPTP dataset (Table 4.2.1)**

| Model | Accuracy | F1 |
|-------|----------|----|
| GCN | 86.25 % | 86.08 % |
| GAT | 85.38 % | 85.34 % |
| GraphSAGE | 86.15 % | 85.92 % |
| SGC | 85.67 % | 85.52 % |
| Cheb | 86.14 % | 85.94 % |
| GIN | 85.93 % | 85.79 % |
| GT | 86.16 % | 86.06 % |
| PC-GCN | 87.77 % | 87.82 % |
| TW-GNN | 88.12 % | 88.04 % |
| **ATWGNNS** | **88.61 %** | **88.45 %** |

**CNF dataset (Table 4.2.2)**

| Model | Accuracy | F1 |
|-------|----------|----|
| GCN | 80.51 % | 80.07 % |
| GAT | 81.80 % | 81.62 % |
| GraphSAGE | 82.96 % | 82.64 % |
| SGC | 80.08 % | 79.95 % |
| Cheb | 81.32 % | 80.80 % |
| GIN | 82.09 % | 81.70 % |
| GT | 82.33 % | 82.01 % |
| PC-GCN | 84.17 % | 83.98 % |
| TW-GNN | 84.24 % | 83.72 % |
| **ATWGNNS** | **84.74 %** | **84.49 %** |

Refer to the paper for the full tables, including the DAGs-vs-S-DAGs comparison (Tables 4.4 / 4.5) and the ablation study (Table 4.6).

## Ablation summary

To switch between the simplified and the original graph representation, replace the `from graph import Graph` import in `dataset.py` with `from graph_DAGs import Graph` (and rebuild the `processed/*.pt` cache). The four ablation variants reported in Table 4.6 correspond to:

| Variant | Graph repr. | GNN |
|---------|-------------|-----|
| ATWGNNS<sub>R−</sub> | DAGs | ATWGNNS |
| ATWGNNS<sub>A−</sub> | S-DAGs | TW-GNN (attention removed) |
| ATWGNNS<sub>AR−</sub> | DAGs | TW-GNN |
| ATWGNNS<sub>T−</sub> | S-DAGs | attention only, no term-walk |
| **ATWGNNS** | **S-DAGs** | **full model** |

## Contact

Issues and pull requests are welcome. For paper-related questions, please contact the corresponding author at `trli@swjtu.edu.cn`.
