import torch
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import f_oneway
import json
import seaborn as sns

def get_feature_activations(sae, H_norm, device):
    sae.eval()

    with torch.no_grad():
        _, Z = sae(H_norm.to(device))

    return Z.cpu()


def compute_feature_importance(Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor: # variance: Does feature differ across classes?
    """
    Compute variance-based class separability for each SAE feature.

    Args:
        Z: Document-level SAE activations [N, F]
        Y: Predicted labels [N]

    Returns:
        Tensor of feature importance scores [F]
    """
    unique_labels = torch.unique(Y)

    feature_stats = {}

    for label in unique_labels:
        mask = (Y == label)
        feature_stats[int(label.item())] = Z[mask].mean(dim=0)

    # multi-class safe: variance across classes
    stacked = torch.stack(list(feature_stats.values()))
    importance = stacked.var(dim=0)

    return importance


def get_top_features(importance: torch.Tensor, k: int = 10) -> torch.Tensor:
    topk = torch.topk(importance, k=k)
    return topk.indices

def get_bottom_features(importance: torch.Tensor, k: int) -> torch.Tensor:
    bottomk = torch.argsort(importance)[:k]
    return bottomk

# def compute_ttest(Z, Y): # Is difference statistically significant?
#     labels = torch.unique(Y)
#     assert len(labels) == 2, "binary only"

#     z0 = Z[Y == labels[0]]
#     z1 = Z[Y == labels[1]]

#     p_values = []

#     for i in range(Z.shape[1]):
#         _, p = ttest_ind(z0[:, i], z1[:, i], equal_var=False)
#         p_values.append(p)

#     return torch.tensor(p_values)

def compute_ttest(Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    labels = torch.unique(Y)

    if len(labels) < 2:
        raise ValueError("Need at least 2 classes")

    p_values = []

    for i in range(Z.shape[1]):
        groups = [Z[Y == label][:, i].numpy() for label in labels]

        _, p = f_oneway(*groups)
        p_values.append(p)

    return torch.tensor(p_values)

def compute_logistic_importance(Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor: # Is feature predictive?
    Z_np = Z.numpy()
    Y_np = Y.numpy()

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Z_np, Y_np)

    importance = np.abs(clf.coef_).mean(axis=0)

    return torch.tensor(importance)

def combine_scores(variance: torch.Tensor, p_values: torch.Tensor, regression_weights: torch.Tensor) -> torch.Tensor:
    """
    Higher score = more important feature
    """

    p_score = -torch.log(p_values + 1e-8)  # smaller p => higher score

    score = (
        variance
        + regression_weights
        + p_score
    )

    return score




def show_top_texts( output_file, feature_idx, Z, dataset, text_col, split="train", k=5):

    values = Z[:, feature_idx]
    top_indices = torch.topk(values, k=k).indices

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n===== Feature {feature_idx} =====\n\n")

        for rank, idx in enumerate(top_indices, 1):
            idx = int(idx)

            f.write(f"[Top {rank}]\n")
            f.write(dataset[split][idx][text_col])
            f.write("\n" + "-"*80 + "\n")



def save_top_texts_json(output_file: str,
    feature_idx: int,
    Z: torch.Tensor,
    dataset,
    text_col: str,
    split: str = "train",
    k: int = 5) -> None:

    """
    Save the top-k highest activating documents for a given SAE feature.

    Args:
        output_file: Path to JSON output file.
        feature_idx: SAE feature index.
        Z: Document-level SAE activations [N, F].
        dataset: HuggingFace dataset object.
        text_col: Name of text column.
        split: Dataset split to inspect.
        k: Number of top examples to save.
    """


    values = Z[:, feature_idx]
    top = torch.topk(values, k=k)

    feature_data = {
        "feature_id": int(feature_idx),
        "examples": []
    }

    for rank, (idx, activation) in enumerate(zip(top.indices, top.values), 1):
        idx = int(idx)

        feature_data["examples"].append({
            "rank": rank,
            "dataset_index": idx,
            "activation": float(activation),
            "text": dataset[split][idx][text_col]
        })

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    except:
        all_data = []

    all_data.append(feature_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

def p_val_histogram(p_values: torch.Tensor) -> None:

    """
    Plot distribution of p-values across SAE features.

    Args:
        p_values: Feature-level p-values.
    """
    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(p_values.numpy(), bins=100)
    ax.axvline(0.05)

    ax.set_xlabel("p-value")
    ax.set_ylabel("Number of Features")
    ax.set_title("Distribution of Feature Significance")

    plt.tight_layout()
    plt.savefig("graphs/pvalue_histogram.png", dpi=300)
    #plt.show()

def var_log_scatter(importance: torch.Tensor, reg_weights: torch.Tensor, rho: float) -> None:
    """
    Plot relationship between variance-based importance
    and logistic regression weights.

    Args:
        importance: Variance-based feature scores.
        reg_weights: Logistic regression coefficients.
        rho: Spearman correlation coefficient.
    """
    fig, ax = plt.subplots(figsize=(8,6))

    idx = np.random.choice(len(importance), 5000, replace=False)

    ax.scatter(
        importance.numpy()[idx],
        reg_weights.numpy()[idx],
        alpha=0.4
    )

    ax.set_xlabel("Variance-Based Importance")
    ax.set_ylabel("Logistic Regression Weight")
    ax.set_title("Cross-Metric Feature Alignment")

    #rho = 0.218
    ax.text(
        0.05, 0.95,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        verticalalignment='top'
    )

    plt.tight_layout()
    plt.savefig("graphs/variance_logistic_scatter.png", dpi=300)
    #plt.show()

def ranking_overlap(values: list[int]) -> None:
    """
    Plot overlap among top-ranked SAE features.

    Args:
        values: Pairwise overlap counts.
    """
    labels = [
        "Variance-Logistic",
        "Variance-Score",
        "Logistic-Score"
    ]

    #values = [16, 16, 50]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(labels, values)

    ax.set_ylabel("Top-100 Feature Overlap")
    ax.set_title("Agreement Across Ranking Criteria")

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("graphs/ranking_overlap.png", dpi=300)
    #plt.show()
