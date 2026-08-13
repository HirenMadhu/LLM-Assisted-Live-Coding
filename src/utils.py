
import torch
from torch.utils.data import Dataset

def center(X):
    return X - X.mean(dim=0, keepdim=True)

def center_gram(X):
    n = X.size(0)
    I = torch.eye(n, device=X.device)
    ones = torch.ones((n, n), device=X.device) / n
    return X - ones @ X - X @ ones + ones @ X @ ones

def gram_linear(X):
    return X @ X.T

def cca(X, Y, reg=1e-4):
    # Center the data
    X = center(X)
    Y = center(Y)

    n = X.size(0)

    # Compute covariance matrices
    C_xx = (X.T @ X) / (n - 1) + reg * torch.eye(X.size(1), device=X.device)
    C_yy = (Y.T @ Y) / (n - 1) + reg * torch.eye(Y.size(1), device=Y.device)
    C_xy = (X.T @ Y) / (n - 1)

    # Cholesky decomposition
    inv_C_xx = torch.linalg.inv(C_xx)
    inv_C_yy = torch.linalg.inv(C_yy)

    # Solve generalized eigenvalue problem
    M = inv_C_xx @ C_xy @ inv_C_yy @ C_xy.T
    eigvals = torch.linalg.eigvalsh(M)  # real-valued eigenvalues
    eigvals = torch.clamp(eigvals, min=0)  # numerical stability

    canonical_corrs = torch.sqrt(eigvals)
    return canonical_corrs.sum()

def cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = gram_linear(X)
    L = gram_linear(Y)

    Kc = center_gram(K)
    Lc = center_gram(L)

    hsic = (Kc * Lc).sum()
    norm_x = torch.norm(Kc)
    norm_y = torch.norm(Lc)
    return hsic / (norm_x * norm_y + 1e-8)


class EmbeddingDataset(Dataset):
    def __init__(self, code_embeddings, wav_embeddings):
        self.code_embeddings = code_embeddings
        self.wav_embeddings = wav_embeddings

    def __len__(self):
        return len(self.code_embeddings)

    def __getitem__(self, idx):
        return self.code_embeddings[idx], self.wav_embeddings[idx]