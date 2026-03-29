# GNN for IC50 Prediction from ChEMBL — Complete Pipeline Plan

## 1. Research Summary & Key Findings

### 1.1 What is IC50 and Why pIC50?

IC50 is the concentration of a compound required to inhibit a biological process by 50%. Raw IC50 values span many orders of magnitude (from picomolar to millimolar), making them difficult to work with directly. The standard practice in the field is to convert to **pIC50 = −log₁₀(IC50 in M)**, which produces a roughly normal distribution in the range 0–10, where higher values indicate more potent compounds.

ChEMBL already provides a pre-computed `pchembl_value` field in the `activities` table that does exactly this transformation. This is defined as −log₁₀(molar value) for IC50, EC50, Ki, Kd, etc., and is only populated for clean, exact-measurement records.

### 1.2 Key Papers & Literature Insights

| Source | Key Takeaway |
|--------|-------------|
| Landrum (2023) — "Hazards of combining IC50 assays" | Combining IC50 from different assays introduces ~0.3–1.0 log-unit noise. Careful curation with `confidence_score = 9`, `SINGLE PROTEIN` targets, and metadata matching is essential. |
| Papadatos et al. (2015) — ChEMBL data curation paper | ChEMBL has a built-in curation workflow: unit standardization, log/−log conversion, data validity flags (`data_validity_comment`), and duplicate detection (`potential_duplicate`). Leverage these fields. |
| Yang et al. (2019) — Chemprop (D-MPNN) | Scaffold splitting is a much better proxy for real-world generalization than random splitting. Bond-level message passing often outperforms atom-level. Baseline GCN with atom features is a strong starting point. |
| BLOPIG Oxford (2022) — SMILES to PyG graph | Reference implementation for atom/bond featurization with RDKit → PyTorch Geometric. One-hot encoding of atom type, degree, charge, hybridization, aromaticity, ring membership. |
| Atz et al. (2021) — Geometric DL on molecules | GCN is the simplest baseline; GAT and GIN are natural next steps. Global mean pooling is standard for graph-level regression. |
| Zdrazil et al. (2024) — systematic study of molecular property prediction | For regression, using pIC50 directly is preferred over classification with arbitrary cutoffs. Activity cliffs (structurally similar molecules with very different potencies) are a fundamental challenge. |

### 1.3 Noise Expectations

Even with careful curation, IC50 data from different labs/assays has an inherent noise floor of ~0.3 log units. An R² of 0.6–0.7 on a multi-target combined dataset with scaffold split is a reasonable baseline; R² > 0.85 is achievable per-target with sufficient data and careful curation.

---

## 2. Data Extraction from ChEMBL PostgreSQL

### 2.1 Key Tables & Their Roles

```
activities          — bioactivity measurements (IC50, pchembl_value, etc.)
assays              — assay metadata (type, confidence_score, tid)
target_dictionary   — target info (target_type, organism, chembl_id)
molecule_dictionary — molecule registry (molregno, chembl_id)
compound_structures — SMILES, InChI, molfiles
compound_properties — precomputed MW, LogP, PSA, HBA, HBD, etc.
molecule_hierarchy  — parent/salt relationships
```

### 2.2 Base Extraction Query (Run in Batches)

```sql
SELECT
    act.activity_id,
    act.molregno,
    md.chembl_id            AS compound_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.data_validity_comment,
    act.potential_duplicate,
    ass.assay_id,
    ass.assay_type,
    ass.confidence_score,
    ass.chembl_id            AS assay_chembl_id,
    td.chembl_id             AS target_chembl_id,
    td.pref_name             AS target_name,
    td.target_type,
    td.organism
FROM activities act
JOIN assays ass             ON act.assay_id = ass.assay_id
JOIN target_dictionary td   ON ass.tid = td.tid
JOIN molecule_dictionary md ON act.molregno = md.molregno
JOIN compound_structures cs ON md.molregno = cs.molregno
WHERE act.standard_type = 'IC50'
  AND act.standard_units = 'nM'
  AND act.standard_relation = '='
  AND act.pchembl_value IS NOT NULL
  AND act.standard_value > 0
  AND (act.data_validity_comment IS NULL
       OR act.data_validity_comment = 'Manually validated')
  AND act.potential_duplicate = 0
  AND ass.assay_type IN ('B', 'F')          -- Binding & Functional only
  AND ass.confidence_score = 9              -- Highest confidence target mapping
  AND td.target_type = 'SINGLE PROTEIN'    -- Avoid complexes/ambiguous targets
ORDER BY act.activity_id
LIMIT 300000 OFFSET {batch_offset};
```

**Why these filters:**
- `standard_relation = '='` — only exact measurements, not `>` or `<` censored values
- `pchembl_value IS NOT NULL` — ChEMBL already validated and computed the −log₁₀ transform
- `data_validity_comment IS NULL` — removes records flagged as "Outside typical range" etc.
- `potential_duplicate = 0` — removes re-cited measurements from other papers
- `assay_type IN ('B','F')` — binding and functional assays are the most relevant for IC50
- `confidence_score = 9` — highest confidence in target assignment
- `SINGLE PROTEIN` — avoids ambiguous multi-component targets

### 2.3 Batched Extraction in Python

```python
import psycopg2
import pandas as pd

BATCH_SIZE = 300_000

conn = psycopg2.connect(
    host="localhost", dbname="chembl_36",
    user="your_user", password="your_password"
)

offset = 0
batch_num = 0

while True:
    query = BASE_QUERY.format(batch_offset=offset)
    df = pd.read_sql(query, conn)
    if df.empty:
        break
    df.to_parquet(f"raw_batches/batch_{batch_num:04d}.parquet", index=False)
    offset += BATCH_SIZE
    batch_num += 1
    print(f"Batch {batch_num}: {len(df)} rows extracted")

conn.close()
```

---

## 3. Data Cleaning Pipeline (Per Batch)

Each batch goes through the following cleaning steps:

### 3.1 SMILES Validation & Standardization

```python
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def standardize_smiles(smiles):
    """Validate, sanitize, and standardize a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Remove fragments (keep largest)
    chooser = rdMolStandardize.LargestFragmentChooser()
    mol = chooser.choose(mol)

    # Uncharge (neutralize)
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)

    # Canonical SMILES
    return Chem.MolToSmiles(mol)
```

**Why fragment removal?** Some ChEMBL entries represent salts (e.g., `drug_molecule.HCl`). We want the parent molecule only. The `molecule_hierarchy` table can also help, but RDKit standardization is more robust.

### 3.2 Duplicate Handling

When multiple measurements exist for the same compound–target pair:

```python
def aggregate_duplicates(df):
    """
    For each (canonical_smiles, target_chembl_id) pair:
    - If multiple pchembl_values, take the median
    - Flag pairs where std > 1.0 (high disagreement) for removal
    """
    grouped = df.groupby(['std_smiles', 'target_chembl_id']).agg(
        pchembl_median=('pchembl_value', 'median'),
        pchembl_std=('pchembl_value', 'std'),
        count=('pchembl_value', 'count')
    ).reset_index()

    # Remove high-variance compound-target pairs
    grouped = grouped[
        (grouped['pchembl_std'].isna()) |   # single measurement
        (grouped['pchembl_std'] <= 1.0)     # within 1 log unit
    ]
    return grouped
```

### 3.3 Molecular Filters

```python
from rdkit.Chem import Descriptors

def is_valid_molecule(mol):
    """Basic drug-likeness and quality filters."""
    if mol is None:
        return False
    mw = Descriptors.ExactMolWt(mol)
    num_atoms = mol.GetNumHeavyAtoms()

    # Exclude tiny fragments and enormous molecules
    if num_atoms < 5 or num_atoms > 100:
        return False
    if mw < 100 or mw > 1000:
        return False

    return True
```

### 3.4 pIC50 Range Filtering

The pchembl_value should typically fall in the range 2–12. Values outside this range are suspicious:

```python
df = df[(df['pchembl_value'] >= 2.0) & (df['pchembl_value'] <= 12.0)]
```

### 3.5 Complete Per-Batch Cleaning Pipeline

```python
def clean_batch(input_path, output_path):
    df = pd.read_parquet(input_path)

    # Step 1: Standardize SMILES
    df['std_smiles'] = df['canonical_smiles'].apply(standardize_smiles)
    df = df.dropna(subset=['std_smiles'])

    # Step 2: Validate molecules
    df['mol'] = df['std_smiles'].apply(Chem.MolFromSmiles)
    df['valid'] = df['mol'].apply(is_valid_molecule)
    df = df[df['valid']].drop(columns=['valid'])

    # Step 3: pIC50 range filter
    df = df[(df['pchembl_value'] >= 2.0) & (df['pchembl_value'] <= 12.0)]

    # Step 4: Aggregate duplicates per compound-target
    df = aggregate_duplicates(df)

    # Step 5: Save
    df.to_parquet(output_path, index=False)

    return len(df)
```

---

## 4. Graph Construction (SMILES → PyG Data Objects)

### 4.1 Node Features (Atom-Level)

Each atom gets a feature vector combining one-hot and scalar features. Based on literature best practices:

| Feature | Encoding | Dimension |
|---------|----------|-----------|
| Atom type (element) | One-hot over [C, N, O, S, F, P, Cl, Br, I, Si, B, Se, Other] | 13 |
| Degree (num heavy neighbors) | One-hot over [0, 1, 2, 3, 4, 5+] | 6 |
| Formal charge | One-hot over [-2, -1, 0, 1, 2, Other] | 6 |
| Hybridization | One-hot over [SP, SP2, SP3, SP3D, SP3D2, Other] | 6 |
| Is aromatic | Binary | 1 |
| Is in ring | Binary | 1 |
| Total num Hs | One-hot over [0, 1, 2, 3, 4+] | 5 |
| **Total per atom** | | **38** |

```python
import numpy as np
from rdkit import Chem

ATOM_LIST = ['C','N','O','S','F','P','Cl','Br','I','Si','B','Se']

def one_hot(val, allowed, extra_category=True):
    vec = [int(val == a) for a in allowed]
    if extra_category:
        vec.append(int(val not in allowed))
    return vec

def atom_features(atom):
    return np.array(
        one_hot(atom.GetSymbol(), ATOM_LIST)          # 13
        + one_hot(atom.GetDegree(), [0,1,2,3,4])      # 6
        + one_hot(atom.GetFormalCharge(), [-2,-1,0,1,2]) # 6
        + one_hot(str(atom.GetHybridization()),
                  ['SP','SP2','SP3','SP3D','SP3D2'])    # 6
        + [int(atom.GetIsAromatic())]                   # 1
        + [int(atom.IsInRing())]                        # 1
        + one_hot(atom.GetTotalNumHs(), [0,1,2,3])     # 5
    , dtype=np.float32)                                 # = 38
```

### 4.2 Edge Features (Bond-Level)

| Feature | Encoding | Dimension |
|---------|----------|-----------|
| Bond type | One-hot [SINGLE, DOUBLE, TRIPLE, AROMATIC] | 4 |
| Is conjugated | Binary | 1 |
| Is in ring | Binary | 1 |
| **Total per bond** | | **6** |

```python
def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ], dtype=np.float32)
```

### 4.3 Full Graph Construction → PyTorch Geometric Data

```python
import torch
from torch_geometric.data import Data

def mol_to_graph(smiles, y_value, target_id=None):
    """Convert SMILES → PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # Edge index & edge features (undirected → add both directions)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bf, bf])

    if len(edge_indices) == 0:
        # Single-atom molecule (shouldn't happen after filtering, but be safe)
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y_value], dtype=torch.float32),
        smiles=smiles,
        target_id=target_id
    )
```

### 4.4 Batch Graph Construction & Saving

```python
from torch_geometric.data import InMemoryDataset
import os

def process_batch_to_graphs(cleaned_parquet, output_dir, batch_id):
    df = pd.read_parquet(cleaned_parquet)
    graphs = []

    for _, row in df.iterrows():
        g = mol_to_graph(
            row['std_smiles'],
            row['pchembl_median'],
            target_id=row.get('target_chembl_id')
        )
        if g is not None:
            graphs.append(g)

    # Save as a list of Data objects
    torch.save(graphs, os.path.join(output_dir, f"graphs_{batch_id:04d}.pt"))
    return len(graphs)
```

---

## 5. Scaffold-Based Train/Val/Test Split

### 5.1 Why Scaffold Split?

Random splitting leaks information: structurally similar molecules end up in both train and test, inflating metrics. Scaffold splitting assigns molecules with the same Bemis-Murcko core to the same set, testing the model's ability to generalize to new chemical scaffolds — which is what matters in real drug discovery.

### 5.2 Implementation

```python
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random

def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "UNKNOWN"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False
    )
    return scaffold

def scaffold_split(smiles_list, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    """
    Split by Bemis-Murcko scaffold.
    Returns lists of indices for train, val, test.
    """
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffolds[generate_scaffold(smi)].append(idx)

    # Sort scaffolds by size (largest first → training set)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train_idx, val_idx, test_idx = [], [], []
    n = len(smiles_list)

    for group in scaffold_sets:
        if len(train_idx) + len(group) <= frac_train * n:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= frac_val * n:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx
```

### 5.3 Applying the Split Across All Batches

Since batching is necessary for memory, the split must be computed **globally** — you need all SMILES to assign scaffolds, but you don't need all features in memory at once:

```python
# Step 1: Collect all (smiles, target) from cleaned parquets — lightweight
all_smiles = []
all_meta = []
for f in sorted(glob("cleaned_batches/*.parquet")):
    df = pd.read_parquet(f, columns=['std_smiles', 'target_chembl_id', 'pchembl_median'])
    all_smiles.extend(df['std_smiles'].tolist())
    all_meta.append(df)

# Step 2: Compute scaffold split on SMILES (just strings → low memory)
train_idx, val_idx, test_idx = scaffold_split(all_smiles)

# Step 3: Create index sets → save as mapping file
split_map = pd.DataFrame({
    'global_idx': range(len(all_smiles)),
    'split': ['train']*len(all_smiles)  # placeholder
})
for i in val_idx: split_map.loc[i, 'split'] = 'val'
for i in test_idx: split_map.loc[i, 'split'] = 'test'
split_map.to_parquet("split_map.parquet")
```

---

## 6. Model Architecture

### 6.1 Baseline GCN (as per teacher's specification)

```
Input (38 atom features)
  ↓
GCNConv(38 → 64) → ReLU
  ↓
GCNConv(64 → 64) → ReLU
  ↓
GCNConv(64 → 64) → ReLU
  ↓
global_mean_pool         (graph-level vector of dim 64)
  ↓
Linear(64 → 32) → ReLU
  ↓
Linear(32 → 1)          (pIC50 prediction)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BaselineGCN(nn.Module):
    def __init__(self, num_node_features=38, hidden_dim=64, head_dim=32):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.lin1 = nn.Linear(hidden_dim, head_dim)
        self.lin2 = nn.Linear(head_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Pooling → graph-level representation
        x = global_mean_pool(x, batch)

        # Regression head
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x.squeeze(-1)
```

### 6.2 Architecture Notes

- **GCNConv** is Kipf & Welling's graph convolution: each layer aggregates neighbor features with degree normalization. 3 layers = receptive field of 3 hops.
- **global_mean_pool** averages all node embeddings in each graph to produce a single vector per molecule. This is permutation-invariant as required.
- **Edge features are NOT used by GCNConv** — this is a known limitation. For a future improvement, switch to `GATConv`, `GINEConv`, or `NNConv` which can incorporate edge attributes.
- The model outputs a single scalar = predicted pIC50.

### 6.3 Why This Architecture is a Good Baseline

The GCN baseline is simple, well-understood, and fast to train. Literature shows that even simple GCNs achieve competitive results on molecular property prediction (within 5–10% of SOTA on many benchmarks). This gives you a solid reference point before exploring:
- **GIN** (Graph Isomorphism Network) — provably more expressive
- **GAT** (Graph Attention Network) — learnable attention over neighbors
- **D-MPNN** (Directed Message Passing) — message passing on bonds, not atoms

---

## 7. Training Pipeline

### 7.1 DataLoader Setup (Batched Training)

```python
from torch_geometric.loader import DataLoader

def load_split_data(graph_files, split_map, split_name):
    """Load graph Data objects for a given split."""
    indices = set(split_map[split_map['split'] == split_name]['global_idx'].tolist())
    graphs = []
    global_idx = 0
    for f in sorted(graph_files):
        batch_graphs = torch.load(f)
        for g in batch_graphs:
            if global_idx in indices:
                graphs.append(g)
            global_idx += 1
    return graphs

# Create DataLoaders
train_data = load_split_data(graph_files, split_map, 'train')
val_data   = load_split_data(graph_files, split_map, 'val')
test_data  = load_split_data(graph_files, split_map, 'test')

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=128, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False)
```

**Note:** PyG's DataLoader handles graph batching automatically — it merges multiple graphs into a single disconnected graph with a `batch` vector that maps each node to its source graph.

### 7.2 Training Loop

```python
from torch.optim import Adam
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

model = BaselineGCN(num_node_features=38)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {
        'rmse': np.sqrt(mean_squared_error(labels, preds)),
        'r2': r2_score(labels, preds),
        'mae': np.mean(np.abs(labels - preds)),
    }
```

### 7.3 Full Training with Early Stopping

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

best_val_rmse = float('inf')
patience = 15
patience_counter = 0

for epoch in range(1, 201):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_metrics = evaluate(model, val_loader, device)

    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
          f"Val RMSE: {val_metrics['rmse']:.4f} | Val R²: {val_metrics['r2']:.4f}")

    if val_metrics['rmse'] < best_val_rmse:
        best_val_rmse = val_metrics['rmse']
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Final evaluation on test set
model.load_state_dict(torch.load('best_model.pt'))
test_metrics = evaluate(model, test_loader, device)
print(f"Test RMSE: {test_metrics['rmse']:.4f} | "
      f"Test R²: {test_metrics['r2']:.4f} | "
      f"Test MAE: {test_metrics['mae']:.4f}")
```

### 7.4 Hyperparameters Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-3 | Standard for Adam |
| Weight decay | 1e-5 | Light regularization |
| Batch size | 64 | Good balance for GNNs |
| Hidden dim | 64 | Teacher's spec, reasonable for baseline |
| GCN layers | 3 | 3-hop receptive field |
| Epochs | Up to 200 | With early stopping (patience=15) |
| Loss | MSE | Standard for regression |
| Optimizer | Adam | Standard |

---

## 8. Evaluation Metrics

| Metric | What It Tells You |
|--------|-------------------|
| **RMSE** (on pIC50) | Average prediction error in log units. < 0.8 is decent for multi-target scaffold split. |
| **R²** | Proportion of variance explained. > 0.5 is reasonable for scaffold split; > 0.7 is good. |
| **MAE** | More robust to outliers than RMSE. |
| **Per-target breakdown** | Some targets will be much easier/harder. Report separately. |

---

## 9. Decision: Single-Target vs Multi-Target

You have two strategies:

**Option A — Single-target model**: Pick one target with lots of data (e.g., EGFR = CHEMBL203, ~10k+ IC50 values). Train a separate GCN for that target. This gives the cleanest data and most interpretable results. Best for a first proof-of-concept.

**Option B — Multi-target model**: Use data across many targets, optionally encoding `target_chembl_id` as a global feature or one-hot. Larger dataset, but more noise. Good for a general-purpose model.

**Recommendation for baseline**: Start with Option A (single target, e.g., EGFR). Once that works, expand to multi-target.

---

## 10. Complete Pipeline Summary

```
┌────────────────────────────────────────────────┐
│  Phase 1: Data Extraction (batched, 300k rows) │
│  SQL → Parquet files                           │
│  Filters: IC50, =, nM, pchembl not null,       │
│           confidence=9, SINGLE PROTEIN,         │
│           no validity flags, no duplicates       │
└──────────────────┬─────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────┐
│  Phase 2: Cleaning (per batch)                 │
│  RDKit standardize → validate → filter         │
│  MW/atom count range → pIC50 range             │
│  Aggregate compound-target duplicates (median) │
└──────────────────┬─────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────┐
│  Phase 3: Scaffold Split (global)              │
│  Collect all SMILES (lightweight)              │
│  Bemis-Murcko scaffold → 80/10/10 split        │
│  Save index → split mapping                    │
└──────────────────┬─────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────┐
│  Phase 4: Graph Construction (per batch)       │
│  SMILES → RDKit Mol → atom/bond features       │
│  → PyG Data(x, edge_index, edge_attr, y)       │
│  Save .pt files                                │
└──────────────────┬─────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────┐
│  Phase 5: Training                             │
│  PyG DataLoader (auto-batches graphs)          │
│  GCNConv×3 → global_mean_pool → MLP → pIC50   │
│  Adam + MSE + early stopping                   │
│  Evaluate: RMSE, R², MAE on test scaffold split│
└────────────────────────────────────────────────┘
```

---

## 11. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| IC50 noise from combining assays | Use `pchembl_value`, `confidence_score=9`, single protein targets, aggregate duplicates with median, flag high-variance pairs |
| Data leakage via random split | Scaffold split enforced |
| Invalid/exotic SMILES | RDKit validation + standardization + molecular filters |
| Oversmoothing in deep GCN | 3 layers is the sweet spot; don't go deeper without residual connections |
| Memory issues | Batch processing at every stage; PyG DataLoader handles graph batching naturally |
| GCN ignores edge features | Acceptable for baseline; upgrade to GINEConv or NNConv later |
