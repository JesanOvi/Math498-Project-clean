from utlis import *
from config import *
from data import *
from model import *
from trainer import *
from sae import *
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW


device = get_device()
datacon = DatasetConfig("/Users/jesanahammed/Desktop/IMDB/IMDB Dataset.csv", "csv", "review", "sentiment", 128)
modelcon = ModelConfig("bert-base-uncased", None)
tokenizer = BertTokenizer.from_pretrained(modelcon.model_name)

loader = DatasetLoader(datacon, tokenizer)
dataset, num_labels, label_map = loader.load()
print(num_labels)

modelcon.num_labels = num_labels

trainingcon = TrainingConfig(32, 1, 2e-5)

train_loader = DataLoader(dataset["train"], batch_size=trainingcon.batch_size)
test_loader = DataLoader(dataset["test"], batch_size=trainingcon.batch_size)

model = BERTClassifier(modelcon, device)
optimizer = AdamW(model.model.parameters(), lr=trainingcon.lr)

trainer = Trainer(model.model, optimizer, device, datacon)

trainer.train(train_loader, trainingcon.epochs)
trainer.plot_loss()
trainer.evaluate(test_loader)
model.save("outputs/models/bert")




