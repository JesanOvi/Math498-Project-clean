import torch
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
import numpy as np


def get_feature_activations(sae, H_norm, device):
    sae.eval()

    with torch.no_grad():
        _, Z = sae(H_norm.to(device))

    return Z.cpu()


def compute_feature_importance(Z, Y): # variance: Does feature differ across classes?
    unique_labels = torch.unique(Y)

    feature_stats = {}

    for label in unique_labels:
        mask = (Y == label)
        feature_stats[int(label.item())] = Z[mask].mean(dim=0)

    # multi-class safe: variance across classes
    stacked = torch.stack(list(feature_stats.values()))
    importance = stacked.var(dim=0)

    return importance


def get_top_features(importance, k=10):
    topk = torch.topk(importance, k=k)
    return topk.indices

def compute_ttest(Z, Y): # Is difference statistically significant?
    labels = torch.unique(Y)
    assert len(labels) == 2, "binary only"

    z0 = Z[Y == labels[0]]
    z1 = Z[Y == labels[1]]

    p_values = []

    for i in range(Z.shape[1]):
        _, p = ttest_ind(z0[:, i], z1[:, i], equal_var=False)
        p_values.append(p)

    return torch.tensor(p_values)

def compute_logistic_importance(Z, Y): # Is feature predictive?
    Z_np = Z.numpy()
    Y_np = Y.numpy()

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Z_np, Y_np)

    importance = np.abs(clf.coef_).mean(axis=0)

    return torch.tensor(importance)

def combine_scores(variance, p_values, regression_weights):
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


def show_top_texts(feature_idx, Z, dataset, text_col, split="train", k=5):
    values = Z[:, feature_idx]

    top_indices = torch.topk(values, k=k).indices

    print(f"\n===== Feature {feature_idx} =====\n")

    for idx in top_indices:
        idx = int(idx)
        print(dataset[split][idx][text_col])
        print("---")