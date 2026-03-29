# ChEMBL GNN — IC50 Prediction Pipeline

Predicts pIC50 from molecular SMILES using a Graph Convolutional Network trained on ChEMBL bioactivity data.

## Pipeline overview

```
01_extract   →   02_clean   →   03_split   →   04_build_graphs   →   05_train
  (SQL)           (RDKit)       (scaffold)        (PyG Data)           (GCN)
```

---

## 1. Requirements

- Python 3.10+
- Docker + Docker Compose
- ~50 GB free disk space for the ChEMBL dump and processed data

---

## 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Database setup

### 3.1 Download the ChEMBL dump

Go to the [ChEMBL release page](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/) and download the PostgreSQL dump for the version you want, e.g.:

```
chembl_35_postgresql.dmp
```

Place it in the `dumps/` directory:

```
dumps/
  chembl_35_postgresql.dmp
```

### 3.2 Configure credentials

```bash
cp .env.example .env
```

Edit `.env` if you want to change the defaults:

```env
CHEMBL_DB_HOST=localhost
CHEMBL_DB_PORT=5432
CHEMBL_DB_NAME=chembl_35
CHEMBL_DB_USER=chembl
CHEMBL_DB_PASSWORD=chembl
```

### 3.3 Start the database

```bash
docker compose up -d
```

The first start automatically restores the ChEMBL dump. **This takes 30–60 minutes** depending on your machine. Watch the progress with:

```bash
docker compose logs -f
```

The database is ready when you see:

```
[init-chembl] Done. ChEMBL database is ready.
```

> To start fresh (e.g. to load a different ChEMBL version), remove the volume:
> ```bash
> docker compose down -v
> ```

---

## 4. Running the pipeline

Always activate the venv first:

```bash
source venv/bin/activate
```

### Step 1 — Extract

Pulls IC50 records from ChEMBL into batched Parquet files under `data/raw/`.

```bash
python scripts/01_extract.py
```

If the run is interrupted, resume from where it left off:

```bash
python scripts/01_extract.py --start-offset 900000
```

### Step 2 — Clean

Validates and standardises SMILES, filters by molecular weight and pIC50 range. Outputs to `data/cleaned/`.

```bash
python scripts/02_clean.py
```

### Step 3 — Split

Assigns each molecule to train/val/test using Bemis-Murcko scaffold splitting. The split map is saved to `data/splits/`.

```bash
python scripts/03_split.py
```

You can generate multiple named splits for benchmarking:

```bash
python scripts/03_split.py --output-name scaffold_split_70_15_15 \
                           --frac-train 0.7 --frac-val 0.15 --frac-test 0.15
```

### Step 4 — Build graphs

Converts cleaned SMILES into PyTorch Geometric `Data` objects and saves them as chunked `.pt` files in `data/graphs/`. This only needs to be run once regardless of how many splits you create.

```bash
python scripts/04_build_graphs.py
```

### Step 5 — Train

Trains the GCN model and prints test-set metrics. The best checkpoint is saved to `data/models/`.

```bash
python scripts/05_train.py
```

To use a specific split or give the run a name:

```bash
python scripts/05_train.py --split-map scaffold_split_70_15_15.parquet \
                           --run-name experiment_01
```

---

## 5. Project structure

```
chembl-gnn/
├── config.py               # All constants and paths — edit here, not in scripts
├── requirements.txt
├── docker-compose.yml
├── docker/
│   └── init-chembl.sh      # DB restore script (runs automatically on first start)
├── dumps/                  # Place your ChEMBL .dmp file here (gitignored)
├── data/                   # All generated data (gitignored)
│   ├── raw/                # Extracted Parquet batches
│   ├── cleaned/            # Cleaned Parquet batches
│   ├── graphs/             # PyG graph chunks + chunk_index.parquet
│   ├── splits/             # Split map Parquet files
│   ├── models/             # Saved model checkpoints
│   └── logs/
├── src/
│   ├── extraction/         # SQL query + batched extractor
│   ├── cleaning/           # SMILES standardisation + batch cleaner
│   ├── features/           # Atom (38-dim) and bond (6-dim) featurisers
│   ├── graph/              # SMILES → PyG Data, chunked storage
│   ├── splitting/          # Scaffold split logic
│   └── training/           # Dataset, trainer, metrics
└── scripts/
    ├── 01_extract.py
    ├── 02_clean.py
    ├── 03_split.py
    ├── 04_build_graphs.py
    └── 05_train.py
```
