# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).
Each team member downloads it locally.

## Where to get it
Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli` (version 1)
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

Downloaded on **2026-10-11** (Kaggle dataset version 1, file `house_sale.csv`,
100,775 rows x 51 columns, 141,882,994 bytes).

## How to place it
Anonymous download with `kagglehub` (in `requirements.txt`), from the repo root:

```bash
python -c "import kagglehub, shutil, os; p = kagglehub.dataset_download('sehriyarmemmedli/binaaz-sale-project'); shutil.copy(os.path.join(p, 'house_sale.csv'), 'data/house_sale.csv')"
```

or download the archive from Kaggle by hand and extract it so the CSV lives at:
```
data/house_sale.csv
```
The path is set once at the top of `src/data_prep.py` (`DATA_PATH`).

## Generated files
`python -m src.run_all` writes `data/split_indices.npz` — the frozen
train/val/test row indices (positions in the cleaned frame). On later runs the
split is recomputed and checked against this file; a mismatch raises an error
(delete the file to re-freeze after an intentional cleaning change).

## Notes / gotchas
- Column names are in **Azerbaijani** (e.g. `Sahə`, `Otaq sayı`, `Mərtəbə`,
  `Təmir`, `Kateqoriya`, …). Keep the UTF-8 encoding when you read the file.
- Each listing (`/items/<id>`) appears several times (repeat scrapes); cleaning
  keeps the latest scrape → 64,454 unique listings.
- **Leakage:** `unit_price` (= price / area) and `total_price` (= price) are
  derived from the target and are dropped in `clean()` (`LEAKAGE_COLUMNS`).
