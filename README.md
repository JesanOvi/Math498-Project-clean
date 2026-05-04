# Transformer Interpretability with BERT + SAE

##  Overview

This project combines **BERT-based text classification** with **mechanistic interpretability using Sparse Autoencoders (SAE)**.

Pipeline:

1. Load and preprocess dataset (IMDB)
2. Train a BERT classifier
3. Generate predictions
4. Extract hidden representations using SAE
5. Compute feature importance
6. Analyze top features and associated texts

---

##  Project Structure

├── main_notebook.ipynb # Main entry point (run everything here)
├── model.py # BERT model wrapper and SAE Implementation
├── trainer.py # Training + evaluation logic
├── config.py # Configuration classes
├── utils.py # Utility functions (device)
├── analysis.py # statistical analysis
├── outputs/ # Saved models (ignored in git)
├── pyproject.toml # Dependencies
├── uv.lock # Locked versions

##  Setup (Using uv - Recommended)

### 1. Install uv

```bash
pip install uv
```


### 2. Clone Repository

git clone https://github.com/JesanOvi/Math498-Project-clean.git
cd Math498-Project-clean.git

### 3. Install dependencies
uv sync

### 4. Setup Jupyter kernel (IMPORTANT)
uv add ipykernel
python -m ipykernel install --user --name=uv-env --display-name "Python (uv-env)"

##   How to run
Open notebook.ipynb, then run cells sequentially