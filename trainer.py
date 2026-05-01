import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Trainer:
    def __init__(self, model, optimizer, device, datasetcon, trainconfig):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = datasetcon
        self.trainconfig = trainconfig
        self.losses = []

    def train(self, loader, epochs):
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
        #plt.savefig(path)
        plt.show()

    def evaluate(self, loader):
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

    #Y = get_bert_predictions(texts)