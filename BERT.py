import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np


class datasetinfo:
    def __init__(self, datasetname, tokenizername, tokenizerlength):
        self.datasetname = datasetname
        self.tokenizername = tokenizername
        self.tokenizerlength = tokenizerlength




def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print("Using device:", device)

def loaddataset():
    dataset = load_dataset("imdb")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset["test"], batch_size=16)
    return train_loader, test_loader


# train_loader, text_loader = loaddataset()
# print(len(train_loader))



model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

def train_bert(model, loader, epochs=1):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader)
        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

#train_bert(model, train_loader, epochs=10) 