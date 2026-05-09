import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

class Trainer:
    """
    Fine-tunes a BERT classifier and provides inference utilities.

    Args:
        model: HuggingFace BERT model for sequence classification
        optimizer: PyTorch optimizer
        device: computation device
        datasetcon: dataset configuration (max length, tokenizer, etc.)
        trainconfig: training configuration (batch size, tokenizer, etc.)
    """

    def __init__(self, model, optimizer, device, datasetcon, trainconfig):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = datasetcon
        self.trainconfig = trainconfig
        self.losses = []

    def train(self, loader: DataLoader, epochs: int) -> None:
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for batch in loader:
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg = total_loss / len(loader)
            self.losses.append(avg)
            print(f"Epoch {epoch}: Loss {avg:.4f}")

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Training Loss")
        plt.savefig("graphs/Bert_loss.png")
        #plt.show()
        plt.close()

    def evaluate(self, loader: DataLoader) -> None:
        self.model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask).logits
                pred = logits.argmax(dim=1).cpu()

                preds.extend(pred)
                labels.extend(batch["label"])

        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:\n", cm)

    def get_bert_predictions(self, texts):
        """
        Returns predicted class labels for a list of texts.

        Args:
            texts: list of raw text strings

        Returns:
            torch.Tensor of predicted class IDs
        """
        preds = []
        self.model.eval()
        for i in range(0, len(texts), self.trainconfig.batch_size):
            batch = texts[i:i+self.trainconfig.batch_size]

            inputs = self.trainconfig.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())

        return torch.tensor(preds)

   