from train  import SCINetinitialization
from util.tools import add_attributes
import json

with open("./config.json", 'r', encoding='utf-8') as file:
    config = json.load(file)


@add_attributes(config)
class Config:
    train: bool = False
    show_results: bool = False
    train_path: str = "dataset/train.csv"
    test_path: str = "dataset/test.csv"
    features: str = "MS"
    target: str = "RUL"
    checkpoints: str = "models/"
    pred_len: int = 1
    individual: bool = False
    enc_in: int = 12
    lradj: int = "6"
    use_gpu : bool = True
    device: int = 0

SCI = SCINetinitialization(Config)
SCI.predict('models/best.pth')