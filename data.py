from datasets import load_dataset

class DatasetLoader:
    def __init__(self, config, tokenizer):
        self.cfg = config
        self.tokenizer = tokenizer

    def load(self):
        # -------------------
        # Load CSV
        # -------------------
        dataset = load_dataset("csv", data_files=self.cfg.path)["train"]

        # split
        dataset = dataset.train_test_split(test_size=0.2)

        text_col = self.cfg.text_column
        label_col = self.cfg.label_column

        # -------------------
        # AUTO LABEL ENCODING (IMPORTANT FIX)
        # -------------------
        unique_labels = sorted(set(dataset["train"][label_col]))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

        def encode_label(example):
            example["label"] = label_map[example[label_col]]
            return example

        dataset = dataset.map(encode_label)

        # -------------------
        # TOKENIZATION
        # -------------------
        def tokenize(example):
            return self.tokenizer(
                example[text_col],
                padding="max_length",
                truncation=True,
                max_length=self.cfg.max_length
            )

        dataset = dataset.map(tokenize, batched=True)

        # -------------------
        # FINAL FORMAT
        # -------------------
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )

        num_labels = len(unique_labels)

        return dataset, num_labels, label_map