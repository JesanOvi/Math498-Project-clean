from utlis import *
from config import *
from data import *
from model import *
from trainer import *

from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW

from analysis import *

from scipy.stats import spearmanr

import argparse
import os

class InterpBert:
    def __init__(self):
        self.device = get_device()
        self.datacon = None
        self.modelcon = None
        self.trainingcon = None
        self.saecon = None
        self.tokenizer = None
        self.loader = None
        self.dataset = None
        self.num_labels = None
        self.label_map = None
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.trainer = None
        self.texts = None
        self.Y = None
        self.Z = None

    def set_dataconfig(self,file_path = "IMDBDataset.csv", file_type = "csv", text_column = "review", label_column = "sentiment", max_length = 128):
        print("Using dataset from: ", file_path)
        self.datacon = DatasetConfig(file_path, file_type, text_column, label_column, max_length)

    def set_modelconfig(self, model_name = "bert-base-uncased", num_labels = None):
        self.modelcon = ModelConfig(model_name, num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(self.modelcon.model_name)
        self.loader = DatasetLoader(self.datacon, self.tokenizer)
        self.dataset, self.num_labels, self.label_map = self.loader.load()
        self.modelcon.num_labels = self.num_labels

    def set_trainingconfig(self, batch_size = 32, epochs = 2, lr = 2e-5):
        self.trainingcon = TrainingConfig(batch_size, epochs, lr, self.tokenizer)
        self.train_loader = DataLoader(self.dataset["train"], batch_size=self.trainingcon.batch_size)
        self.test_loader = DataLoader(self.dataset["test"], batch_size=self.trainingcon.batch_size)
    
    def set_saeconfig(self, model_name="gpt2", sae_release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", layer=8, batch_size=4):
        self.saecon = SAEConfig(model_name, sae_release, sae_id, layer, batch_size, None, self.device)
    
    def set_all_config(self):
        self.device = get_device()
        self.set_dataconfig()
        self.set_modelconfig()
        self.set_trainingconfig()
        self.set_saeconfig()
    
    def init_model(self):
        #self.set_all_config()
        self.model = BERTClassifier(self.modelcon, self.device)
        optimizer = AdamW(self.model.model.parameters(), lr=self.trainingcon.lr)
        self.trainer = Trainer(self.model.model, optimizer, self.device, self.datacon, self.trainingcon)

    def fine_tune_model(self):
        self.init_model()
        self.trainer.train(self.train_loader, self.trainingcon.epochs)
        self.trainer.plot_loss()
        self.trainer.evaluate(self.test_loader)
        os.makedirs("outputs/models", exist_ok=True)
        self.model.save("outputs/models/bert")
        print("Model Saved at outputs/models/bert")

    def load_saved_model(self):
        self.init_model()
        if self.model is None:
            self.model = BERTClassifier(self.modelcon, self.device)
        self.model.load("outputs/models/bert")

        self.trainer.model = self.model.model
        self.model.model.eval()

        # print(self.model.model)
        # print(self.trainer.model)
        print(self.model.model is self.trainer.model)
    
    def get_model_prediction(self, num_samples = 5000):
        self.dataset.set_format(type=None)  
        N = num_samples
        self.texts = self.dataset["test"][self.datacon.text_column][:N]
        true_labels = torch.tensor(self.dataset["test"]["label"][:N])
        self.saecon.texts = self.texts
        print("True Lables", true_labels)
        self.Y = self.trainer.get_bert_predictions(self.texts)
        print("Model Predicted labels", self.Y)
    
    def get_sae(self):
        saeobj = ModelWithSAE(self.saecon)
        self.Z = saeobj.compute_sae()
        print("Shape of SAE", self.Z.shape)

    def overlap(self, a, b):
        return len(set(a.tolist()) & set(b.tolist()))
    
    def compute_state(self, num_samples):
        print("Analysis will be done for ", num_samples, "from test set\n")
        self.get_model_prediction(num_samples)
        self.get_sae()
        var = compute_feature_importance(self.Z, self.Y)
        p_values = compute_ttest(self.Z, self.Y)
        p_val_histogram(p_values)
        reg_weights = compute_logistic_importance(self.Z, self.Y)
        var_log_scatter(var, reg_weights)
        significant_features = (p_values < 0.05).sum().item()
        print("Number of significant features", significant_features)
        score = combine_scores(var, p_values, reg_weights)

        k = 100  
        top_var = torch.topk(var, k=k).indices
        top_log = torch.topk(reg_weights, k=k).indices
        top_score = torch.topk(score, k=k).indices
        ov_var_log = self.overlap(top_var, top_log)
        ov_var_score = self.overlap(top_var, top_score)
        ov_log_score = self.overlap(top_log, top_score)
        print("Top", k, "overlap between ranking method: \n")
        print("Variance-Logistic, Variance-Score and Logistic-Score\n")
        print(ov_var_log, ov_var_score, ov_log_score)
        values = []
        values.append(ov_var_log)
        values.append(ov_var_score)
        values.append(ov_log_score)
        ranking_overlap(values)
        rho, p = spearmanr(var.numpy(), reg_weights.numpy())
        print("Spearman Correlation between variance and logistic weights \n")
        print(rho, p)

        top_features = get_top_features(score, k=10)
        print("Top Strong features:", top_features)
        self.dataset.set_format(type=None) 
        print("Documents associated with each strong features are saved in strong_post.txt")
        for f in top_features:
            #print(dataset["train"].column_names)
            show_top_texts("strong_post.txt", f, self.Z, self.dataset, text_col=self.datacon.text_column)

        weak_features = get_bottom_features(score, 10)
        print("Weak Features", weak_features)
        self.dataset.set_format(type=None) 
        print("Documents associated with each weak features are saved in weak_post.txt")
        for f in weak_features:
            #print(dataset["train"].column_names)
            show_top_texts("weak_post.txt", f, self.Z, self.dataset, text_col=self.datacon.text_column)







def build_parser():
    parser = argparse.ArgumentParser(
        description="Transformer Interpretability with BERT + SAE"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "load", "analyze", "full"],
        default="full",
        help="""
        train   -> fine-tune BERT
        load    -> load saved model
        analyze -> run SAE analysis on saved model
        full    -> load model + analyze
        """
    )

    # Dataset config
    parser.add_argument("--data", type=str,
                        default="IMDBDataset.csv")

    parser.add_argument("--text-col", type=str, default="review")
    parser.add_argument("--label-col", type=str, default="sentiment")
    parser.add_argument("--max-length", type=int, default=128)

    # Model config
    parser.add_argument("--model-name", type=str,
                        default="bert-base-uncased")

    # Training config
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)

    # SAE config
    parser.add_argument("--sae-model", type=str, default="gpt2")
    parser.add_argument("--sae-release", type=str,
                        default="gpt2-small-res-jb")
    parser.add_argument("--sae-id", type=str,
                        default="blocks.8.hook_resid_pre")
    parser.add_argument("--layer", type=int, default=8)

    parser.add_argument("--samples", type=int, default=5000)

    return parser


def configure_model(args):
    ob = InterpBert()
    ob.set_all_config()

    ob.set_dataconfig(
        file_path=args.data,
        text_column=args.text_col,
        label_column=args.label_col,
        max_length=args.max_length
    )

    ob.set_modelconfig(
        model_name=args.model_name
    )

    ob.set_trainingconfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )

    ob.set_saeconfig(
        model_name=args.sae_model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        layer=args.layer
    )

    return ob


def main():
    parser = build_parser()
    args = parser.parse_args()

    ob = configure_model(args)

    if args.mode == "train":
        ob.fine_tune_model()

    elif args.mode == "load":
        ob.load_saved_model()

    

    elif args.mode == "full":
        ob.load_saved_model()
        ob.compute_state(num_samples=args.samples)


if __name__ == "__main__":
    main()




