from dataclasses import dataclass

@dataclass
class DatasetConfig:
    def __init__(self, path, type, text_column, label_column, max_lenght):
        self.path = path
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_lenght
        self.type = type
        self.name = "huggingface"

@dataclass
class ModelConfig:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels


@dataclass
class TrainingConfig:
    def __init__(self, batch_size, epochs, lr, tokenizer):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.tokenizer = tokenizer

# @dataclass
# class SAEConfig:
#     input_dim: int
#     hidden_dim: int
#     lambda_l1: float
#     batch_size: int
#     epochs: int
#     model: any
#     loader: any
#     max_sample: int
#     device: any

@dataclass
class gemmaConfig:
    gemma_name: any
    rel: any
    sae_id: any
    batch_size: int
    texts: any
    device: any




@dataclass
class SAEConfig:
    model_name: str
    sae_release: str
    sae_id: str
    layer: int
    batch_size: int
    texts: list
    device: str