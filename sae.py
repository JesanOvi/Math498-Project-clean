import torch.nn as nn
import torch.nn.functional as F

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class SparseAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim)
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim)
        self.lambda_l1 = config.lambda_l1

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z



class hiddenstates:
    def __init__(self, device, model, loader, max_sample):
        self.model = model
        self.loader = loader
        self.max_sample = max_sample
        self.device = device

    def extract_hidden_states(self):
        self.model.eval()

        all_h = []
        all_labels = []
        count = 0

        with torch.no_grad():
            for batch in tqdm(self.loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                last_hidden = outputs.hidden_states[-1]      # (B, seq, dim)
                cls_vectors = last_hidden[:, 0, :]           # (B, dim)

                all_h.append(cls_vectors.cpu())
                all_labels.append(labels.cpu())

                count += cls_vectors.size(0)
                if count >= self.max_sample:
                    break

        H = torch.cat(all_h)
        Y = torch.cat(all_labels)

        return H, Y
    
    def normalize(self, H):
        mean = H.mean(dim=0)
        std = H.std(dim=0)
        std = torch.clamp(std, min=1e-6)

        H_norm = (H - mean) / std
        return H_norm
    

class trainsae:
    def __init__(self, config):
        self.config = config
    
    def train_sae(self):
        lossess = []
        sae = SparseAutoencoder(self.config).to(self.config.device)
        hs = hiddenstates(self.config.device, self.config.model.model, self.config.loader, self.config.max_sample)
        H, Y = hs.extract_hidden_states()
        H_norm = hs.normalize(H)


        dataset = TensorDataset(H_norm)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

        for epoch in range(self.config.epochs):
            total_loss = 0

            for (x,) in tqdm(loader):
                x = x.to(self.config.device)

                optimizer.zero_grad()

                x_hat, z = sae(x)

                recon_loss = F.mse_loss(x_hat, x)
                l1_loss = z.abs().mean()

                loss = recon_loss + self.config.lambda_l1 * l1_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss/len(loader)
            lossess.append(avg_loss)
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        plt.plot(lossess)
        plt.title("Training Loss for SAE")
        plt.show()
        return sae, H_norm, Y
    