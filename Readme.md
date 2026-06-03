# EvolveGCN-T: Self-Attention for Weight Evolution in Dynamic Graphs

Stanford 229 Project
Victoria Yang, Veronica Wang, Kaci Morris

This repository extends the original [IBM EvolveGCN](https://github.com/IBM/EvolveGCN) codebase with **EvolveGCN-T**, a variant that replaces EvolveGCN's recurrent weight evolution with a Transformer encoder that self-attends over the sequence of past GCN weight matrices.

We reproduce the original baselines (EvolveGCN-O and EvolveGCN-H) on three tasks and compare them against our Transformer variant (EvolveGCN-T).

| Variant | Temporal module | Uses node embeddings? |
|---|---|---|
| `egcn_o` | Recurrent (GRU on weights) | No (weight history only) |
| `egcn_h` | Recurrent (GRU + embedding summary) | Yes |
| `egcn_t` **(ours)** | Transformer self-attention over weight history | No (Transformer analog of `egcn_o`) |

The controlled comparison is **`egcn_o` vs `egcn_t`** (identical inputs and config,
only the temporal module differs). `egcn_h` is a more-informed reference baseline.

---

## What we changed/added

| Path | Description |
|---|---|
| `egcn_t.py` | Our EvolveGCN-T module. Copied from `egcn_o.py` but swaps the recurrent `mat_GRU_cell` for a Transformer cell that attends over the weight-state sequence. |
| `run_exp.py` | Added an `egcn_t` branch to `build_gcn(...)`|
| `experiments/transformers/parameters_*_egcn_t.yaml` | EvolveGCN-T configs (copies of the `egcn_o` configs + the `egcn_t_*` hyperparameters). |

---

## Key Files

| File | Function |
|---|---|
|run_exp.py | runs all experiements|
|egcn_o.py | EvolveGCN-O (recurrent, weights only)
|egcn_h.py | EvolveGCN-H (recurrent + node-embedding summary)
|egcn_t.py | EvolveGCN-T|
|log_analyzer.py  |  Analyzes log files and give best-validation-epoch metrics|
| experiments/ | YAML configs (one per dataset per model)|
|data/ | datasets |
|log/ | run logs (auto-named: log/log_<dataset>_<task>_<model>_<ts>_r0.log)|

---

## Setup

### 1. Create an environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

Install PyTorch matching your CUDA version first (we used PyTorch 2.x with CUDA on
an NVIDIA T4), then the rest:

```bash
pip install --upgrade pip
pip install torch
pip install pyyaml numpy pandas scipy scikit-learn wandb
```

### 3. Compatibility fixes

The original code uses NumPy aliases that were removed in NumPy >= 1.24. We had to apply this:

```bash
sed -i -E 's/np\.(float|int|bool|object|str|complex)\b/\1/g' *.py
sed -i -E 's/np\.long\b/int/g; s/np\.unicode\b/str/g' *.py
```

Another issue we encountered:
- Set `save_node_embeddings: False` in any config you run (the embedding-dump path
  is broken and unnecessary for our experiments).

---

## Data setup

All datasets go under `data/`.

**SBM** (synthetic, ships with the repo): the adjacency file
`sbm_50t_1000n_adj.csv` is included in `data/`

**Bitcoin-OTC / Bitcoin-Alpha** (from SNAP):
```bash
cd data
wget https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
wget https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz
gunzip soc-sign-bitcoinotc.csv.gz soc-sign-bitcoinalpha.csv.gz
```

**Elliptic** (Bitcoin transaction graph): download the Elliptic Data Set and place the three CSVs where the elliptic config's `folder:` points:
```
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

---

## How to run

Every experiment is one command: pick a config; the `model:` field inside it selects
`egcn_o` / `egcn_h` / `egcn_t`.

```bash
SEED=1 PYTHONWARNINGS=ignore python run_exp.py \
    --config_file ./experiments/<config>.yaml 2>&1 | tee log/<name>.log
```

- We found that `SEED=N` makes the run reproducible.

### Reproduce the baselines

```bash
SEED=1 PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/parameters_sbm_egcn_o.yaml
SEED=1 PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/parameters_sbm_egcn_h.yaml

SEED=1 PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/parameters_bitcoin_otc_edgecls_egcn_h.yaml
SEED=1 PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/parameters_bitcoin_alpha_edgecls_egcn_h.yaml

SEED=1 PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/parameters_elliptic_egcn_o.yaml
```

### Run our model

```bash
SEED=1 PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/transformers/parameters_sbm_egcn_t.yaml
SEED=1 PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/transformers/parameters_bitcoin_otc_edgecls_egcn_t.yaml
```

### Multi-seed (We found running this helped stablize, SBM seesm to be more stable than Bitcoin)

```bash
for s in 1 2 3; do
  SEED=$s PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/parameters_sbm_egcn_o.yaml
  SEED=$s PYTHONWARNINGS=ignore python run_exp.py --config_file ./experiments/transformers/parameters_sbm_egcn_t.yaml
done
```

### Attention-span ablation for EvolveGCN-T and Bitcoin-OTC

Sweep the attention window `egcn_t_window`:
```bash
for w in 2 3 5 8 10; do
  sed -i -E "s/^([[:space:]]*)egcn_t_window:.*/\1egcn_t_window: ${w}/" \
    experiments/transformers/parameters_bitcoin_otc_edgecls_egcn_t.yaml
  SEED=1 PYTHONWARNINGS=ignore python run_exp.py \
    --config_file ./experiments/transformers/parameters_bitcoin_otc_edgecls_egcn_t.yaml
```

### History length comparison between `egcn_o` and `egcn_t`

We changed `num_hist_steps` for both models:
```bash
for h in 2 10; do
  sed -i -E "s/^([[:space:]]*)num_hist_steps:.*/\1num_hist_steps: ${h}/" \
    experiments/parameters_bitcoin_otc_edgecls_egcn_o.yaml
  SEED=1 PYTHONWARNINGS=ignore python run_exp.py \
    --config_file ./experiments/parameters_bitcoin_otc_edgecls_egcn_o.yaml

  sed -i -E "s/^([[:space:]]*)num_hist_steps:.*/\1num_hist_steps: ${h}/" \
    experiments/transformers/parameters_bitcoin_otc_edgecls_egcn_t.yaml
  sed -i -E "s/^([[:space:]]*)egcn_t_window:.*/\1egcn_t_window: ${h}/" \
    experiments/transformers/parameters_bitcoin_otc_edgecls_egcn_t.yaml
  SEED=1 PYTHONWARNINGS=ignore python run_exp.py \
    --config_file ./experiments/transformers/parameters_bitcoin_otc_edgecls_egcn_t.yaml
done
```

---

## Hyperparameters (found under `gcn_parameters:` in the `egcn_t` configs):

| Hyperparameter | What it is | Default value |
|---|---|---|
| `egcn_t_window` | attention span over the weight history | 10 |
| `egcn_t_heads` | number of attention heads (fallback 1) | 4 |
| `egcn_t_layers` | number of Transformer encoder layers | 1 |
| `egcn_t_dropout` | dropout in the encoder | 0.0 |
| `egcn_t_residual` | add residual to `W_{t-1}` | true |

---

## Analyzing results

Autonamed logs are saved to `log/log_<dataset>_<task>_<model>_<ts>_r0.log`.
Analyze it with the log analyzer:

```bash
python log_analyzer.py log/log_sbm50_link_pred_egcn_o_<ts>_r0.log
```

It prints a CSV row at the best-validation epoch. The metrics that are relevant:

| Task | Relevant Metrics |
|---|---|
| Node classification (Elliptic) | target-class (illicit) F1 |
| Edge classification (Bitcoin) | micro-avg F1 (the `AVG-F1` column) |
| Link prediction (SBM) | MAP |

---

## Our Results

| Dataset | Task | Metric | egcn_o | egcn_h | egcn_t (ours) |
|---|---|---|---|---|---|
| SBM | link prediction | MAP (3 seeds) | **0.194±0.001** | 0.176±0.011 | 0.130±0.029 |
| Bitcoin-OTC | edge classification | micro-F1 (best) | 0.699 | **0.892** | 0.783 |
| Bitcoin-Alpha | edge classification | micro-F1 (best) | 0.637 | 0.856 | — |
| Elliptic | node classification | illicit-F1 | 0.578 | — | — |

Summary: in the matched `egcn_o` vs `egcn_t` comparison, the Transformer loses on
SBM but wins on Bitcoin-OTC (we also tested at history-length 2: 0.746 vs 0.699). 
`egcn_h` is best overall but is more-informed, so it is not a fair and controlled
comparison for `egcn_t`.

---

## Note

The EvolveGCN-O family collapses stochastically. We saw that a run either trains for 
hundreds of epochs or early-stops within ~10 epochs at a degenerate solution. 
To counter this, we use multiple seeds and check `best_epoch` in the logs. 

## Acknowledgments

Base code: [IBM/EvolveGCN](https://github.com/IBM/EvolveGCN).