# Transformer Interpretability with BERT + SAE

##  Overview

This project combines **BERT-based text classification** with **mechanistic interpretability using Sparse Autoencoders (SAE)**.

Pipeline:

1. Load and preprocess dataset (IMDB)
2. Fine-tue a BERT classifier
3. Generate predictions using unseen test dataset
4. Pass same test dataset to GPT-2 and Extract hidden representations using SAE
5. Compute feature importance
6. Analyze top features and associated texts

---

---

## Project Writeup

**Full Report (Main Paper):**  
    [Math498 Project Writeup (PDF)](docs/Writeup.pdf)

This document contains the complete methodology, experiments, results, and analysis for this project, including:

- BERT fine-tuning setup  
- SAE-based feature decomposition  
- Statistical feature importance analysis  
- Cross-model interpretability results  
- Qualitative feature interpretation  

---

## Project Structure

```text
Math498-Project/
├── docs                     
      ├── Writeup.pdf           # Final report of the project
      ├── Math498-Project.zip   # Overleaf Project of the writeup
├── notebook.ipynb              # Main entry point
├── main.py                     # CLI entry point
├── models                
      ├── model.py              # BERT model wrapper + SAE implementation
      ├── trainer.py            # Training and evaluation logic
      ├── data.py               # Dataset loading and preprocessing
├── State
      ├── analysis.py           # Statistical analysis

├── top_posts
      ├── strong_features.json  # contains top 5 documents correspond to each strong features 
      ├── weak_features.json    # contains top 5 documents correspond to each weak features 
      ├── qualitative_analysis_posts.pdf           # contains documents associate with strong and weak features, used to conduct the qualitative analysis.
├── graphs                      # contains all the graphs generated while executing the entire pipeline
├── config.py                   # Configuration classes
├── utils.py                    # Utility functions
├── outputs/                    # Saved models (ignored by git)
├── pyproject.toml              # Dependencies
└── uv.lock                     # Locked dependency versions
```

## Codebase Map

### 1. BERT Classification
Purpose: Train classifier and generate predictions

```
trainer.py -- Fine-tune BERT model, eveluation, get prediction on unseen data.
model.py -- BERT architecture wrapper
main.py -- Calls training pipeline (--mode train)
```

### 2. Representation Extraction (GPT-2 + SAE)
Purpose: Extract transformer residuals and map to sparse features

```
model.py -- Loads GPT-2, SAE forward pass, Calls SAE on residual stream
config.py -- Defines all configurations
```

### 3. Statistical Feature Analysis
Purpose: Identify SAE features aligned with classifier behavior

```
analysis.py -- All statistical analysis e.g. Variance-based ranking, t-test, regression, score calculation
```

### 4. Qualitative Feature Interpretation
Purpose: Interpret top and weak SAE features using real text examples

```
strong_post.txt and weak_post.txt -- contains top 5 documents correspond to each features. This could change everytime the pipeline is run.
combine.pdf -- contians top 5 documents associate with each features for both strong and weak features, used for analysis and include in the report.
```

### 4. Final Report
 ```
 Math498-project.pdf --  contains the entire writeup for the project.
 ```


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

### 5. How to run
Open notebook.ipynb, then run cells sequentially, if you prefer to see the output. Otherwise main.py provide CLI interface to run, configure various objects.

Default dataset path:
```text
IMDBDataset.csv
```
Fine tune BERT, Train a new model:
```bash
uv run main.py --mode train
```
This will fine-tune the BERT model using default dataset and save the model.

Load Saved Model
```bash
uv run main.py --mode load
```

This will load the saved model.

Run Full Interpretability Analysis
Loads saved model + computes SAE feature statistics:
```bash
uv run main.py --mode full
```
Note: GitHub Repo doesn't contain the fine-tuned model (due to the model size). Directly execute uv run main.py --mode full will load null model. It is recommendated to fine-tune the model first (uv run main.py --mode train). If you want to download the fine-tune model follow the link. [fine-tunned BERT] (https://drive.google.com/drive/folders/1ZjG6-KnGj3rBFLWtit-1cdsWs2jH6Hqz?usp=sharing). Download the folder and copy-paste into main project folder. Expected path will be project_main_folder\outputs\models\bert.

Specify sample size:
```bash
uv run main.py --mode full --samples 1000
```
Useful if you want to test the entire pipeline using small sample first.

Use Custom Dataset

```bash
uv run main.py \
    --mode full \
    --data my_dataset.csv \
    --text-col text \
    --label-col label
```

## CLI Arguments

| Argument | Default | Description |
|---------|---------|-------------|
| `--mode` | full | train / load / full |
| `--data` | IMDBDataset.csv | Dataset path |
| `--text-col` | review | Text column |
| `--label-col` | sentiment | Label column |
| `--max-length` | 128 | Tokenization length |
| `--model-name` | bert-base-uncased | HuggingFace model |
| `--batch-size` | 32 | Training batch size |
| `--epochs` | 2 | Number of epochs |
| `--lr` | 2e-5 | Learning rate |
| `--samples` | 5000 | Number of test samples for analysis |
| `--layer` | 8 | Transformer layer for SAE |

---

## Analysis Outputs

The interpretability pipeline computes:

### Feature Variance
Measures feature activation differences across classes

### Statistical Significance
Welch’s t-test for discriminative latent features

### Logistic Importance
Predictive strength of each SAE feature

### Ranking Overlap
Agreement across feature importance methods

### Spearman Correlation
Correlation between statistical and predictive rankings

### Feature Inspection
Displays strongest and weakest latent features with associated text examples

---

## Example

```bash
uv run main.py --mode full --samples 500
```

Output:

```text
Analysis will be done for 500 samples

Shape of SAE torch.Size([500, 24576])

Top 100 overlap between ranking methods:
Variance-Logistic: 41
Variance-Score: 53
Logistic-Score: 48

Spearman Correlation:
0.72
```

---

## Model Configuration

Default BERT setup:

| Parameter | Value |
|----------|-------|
| Model | bert-base-uncased |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Epochs | 2 |
| Sequence Length | 128 |

---

## Notes

- Saved models are stored in:

```text
outputs/models/bert
```

- Large model artifacts are excluded from GitHub

- For small sample sizes, analysis may fail if predictions collapse to a single class

Recommended:

```bash
--samples 500+
```

---

## Author

Jesan Ahammed Ovi  
PhD Student, Computer Science  
Colorado School of Mines



