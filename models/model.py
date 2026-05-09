from transformers import BertForSequenceClassification
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sae_lens import SAE

class BERTClassifier(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=self.cfg.num_labels
        ).to(device)

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)



class GemmaScope(nn.Module):
    def __init__(self, config):
        self.config = config
        self.gemma_tokenizer = None
        self.gemma = None
        self.sae = None
    
    def load_gemma(self):
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(self.config.gemma_name)
        self.gemma = AutoModelForCausalLM.from_pretrained(
            self.config.gemma_name,
            torch_dtype=torch.float16 if self.config.device != "cpu" else torch.float32
        ).to(self.config.device)

        self.gemma.eval()
     
    
    def load_gemmascope(self):
        self.sae, cfg_dict, sparsity = SAE.from_pretrained(release=self.config.rel, sae_id=self.config.sae_id)
        self.sae.to(self.config.device)
        self.sae.eval()
    
    def get_gemma_residuals(self):
        residuals = []

        for i in range(0, len(self.config.texts), self.config.batch_size):
            batch = self.config.texts[i:i+self.config.batch_size]

            inputs = self.gemma_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.config.device)

            with torch.no_grad():
                outputs = self.gemma(
                    **inputs,
                    output_hidden_states=True
                )

            h = outputs.hidden_states[20]

            pooled = h.mean(dim=1)

            residuals.append(pooled.cpu())

        return torch.cat(residuals)
    
    def compute_sae_activation(self):
        self.load_gemma()
        self.load_gemmascope()
        H = self.get_gemma_residuals()
        with torch.no_grad():
            Z = self.sae.encode(H.to(self.config.device)).cpu()

        return Z

#H = get_gemma_residuals(texts)


import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE


# class ModelWithSAE(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#     def load_model(self):
#         #self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.model_name,
#             torch_dtype=torch.float16 if self.config.device != "cpu" else torch.float32
#         ).to(self.config.device)
#         self.model.eval()

#     def load_sae(self):
#         # IMPORTANT: this MUST be a valid SAE release
#         self.sae, _, _ = SAE.from_pretrained(
#             release=self.config.sae_release,
#             sae_id=self.config.sae_id
#         )
#         self.sae.to(self.config.device)
#         self.sae.eval()

#     def get_residuals(self):
#         residuals = []

#         for i in range(0, len(self.config.texts), self.config.batch_size):
#             batch = self.config.texts[i:i+self.config.batch_size]

#             tokens = self.tokenizer(
#                 batch,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=128
#             ).to(self.config.device)

#             with torch.no_grad():
#                 outputs = self.model(**tokens, output_hidden_states=True)

#             # BERT/GPT-style hidden states
#             h = outputs.hidden_states[self.config.layer]  # [B, T, D]

#             # token-level flatten
#             h = h.reshape(-1, h.shape[-1])  # [B*T, D]

#             residuals.append(h.cpu())

#         return torch.cat(residuals)

#     def compute_sae(self):
#         self.load_model()
#         self.load_sae()

#         H = self.get_residuals().to(self.config.device)

#         with torch.no_grad():
#             Z = self.sae.encode(H)

#         return Z.cpu()

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE


class ModelWithSAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    # ----------------------------
    # Load BERT/GPT model
    # ----------------------------
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

       
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device != "cpu" else torch.float32
        ).to(self.config.device)

        self.model.eval()

    # ----------------------------
    # Load pretrained SAE
    # ----------------------------
    def load_sae(self):
        self.sae, _, _ = SAE.from_pretrained(
            release=self.config.sae_release,
            sae_id=self.config.sae_id
        )
        self.sae.to(self.config.device)
        self.sae.eval()

    # ----------------------------
    # Extract residual streams
    # KEEP document boundaries
    # ----------------------------
    def get_residuals(self):
        """
        Returns:
            residuals_per_doc: List[Tensor[T, D]]
        """
        residuals_per_doc = []

        for i in range(0, len(self.config.texts), self.config.batch_size):
            batch = self.config.texts[i:i + self.config.batch_size]

            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.config.device)

            with torch.no_grad():
                outputs = self.model(**tokens, output_hidden_states=True)

            # [B, T, D]
            h = outputs.hidden_states[self.config.layer]

            B = h.shape[0]

            # KEEP each document separate (IMPORTANT)
            for j in range(B):
                residuals_per_doc.append(h[j].detach().cpu())  # [T, D]

        return residuals_per_doc

    # ----------------------------
    # Run SAE + aggregate per doc
    # ----------------------------
    def compute_sae(self):
        self.load_model()
        self.load_sae()

        residuals_per_doc = self.get_residuals()

        Z_docs = []

        with torch.no_grad():
            for h in residuals_per_doc:
                # SAE expects [tokens, d_model]
                z = self.sae.encode(h.to(self.config.device))  # [T, features]

                # aggregate tokens → document representation
                z_doc = z.mean(dim=0)  # [features]

                Z_docs.append(z_doc.cpu())

        return torch.stack(Z_docs)  # [num_docs, features]