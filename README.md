# Clinically-Agnostic-Model

## Dataset (Zenodo)

The datasets are hosted in a compressed archive on **Zenodo** under DOI [`10.5281/zenodo.22998650`](https://doi.org/10.5281/zenodo.22998650) and consist of:

* `dataset_original_public.csv`: 28 raw, anonymized attributes from health insurance data.
* `dataset_model_enrichment_public.csv`: 92 feature-engineered attributes used directly for model training.

> **Note:** The execution script automatically streams the archive temporarily and extracts only the specific file required for training. 

## How to Run

### 1. Clone the Repository
Download or Clone the Repository.
Download the repository as a ZIP archive from this page (or clone using the anonymized URL), then navigate to the project folder.

### 2. Create and Activate a Virtual Environment
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the model
python MDL_public_CB_enrichment.py
