from DataProcessing.data import gen_data
from train import SCINetinitialization
from util.tools import add_attributes

try:
    from hyperopt import hp
    from hyperopt.pyll.stochastic import sample
except ImportError:
    print(
        "In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt), unless you make get_params() use something else."
    )
#


# handle floats which should be integers
# works with flat params
def handle_integers(params):

    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v

    return new_params





base_config_dict: dict = {
    "train": False,
    "show_results": False,
    "train_path": "dataset/train.csv",
    "test_path": "dataset/test.csv",
    "features": "MS",
    "target": "RUL",
    "checkpoints": "models/",
    "pred_len": 1,
    "individual": False,
    "enc_in": 12,
    "lradj": "6",
    "use_gpu": True,
    "device": 0,
}


@add_attributes(base_config_dict)
class Config:
    seq_len: int = 32
    batch_size: int = 32
    learning_rate: float = 0.001
    l1: int = 32
    cnn_kernel_size: int = 15
    loss: str = "mse"
    train_epochs: int = 100
    D_kernel_size: int = 25


space: dict = {
    "seq_len": hp.quniform("seq_len", 16, 64, 2),
    "batch_size": hp.quniform("batch_size", 8, 128, 8),
    # "batch_size": hp.choice("batch_size",[8, 16, 32, 64, 128]),
    "learning_rate": hp.uniform("learning_rate", 0.0001, 0.002),
    "l1": hp.quniform("l1", 32, 64, 4),
    "cnn_kernel_size": hp.quniform("cnn_kernel_size", 10, 25, 1),
    "loss": hp.choice("loss", ["mse", "mae"]),
    # "PCA_n": hp.quniform("PCA_n", 8, 13, 1),
    # "threshold": hp.quniform("threshold", 1.2, 2.7, 0.1),
    # "max_RUL": hp.quniform("max_RUL", 110, 150, 5),
    "D_kernel_size": hp.choice(
        "D_kernel_size", [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    ),
}


def get_params():
    params = sample(space)
    params = handle_integers(params)
    return params


def try_params(n_iterations, params):
    print("Trying params: " + str(params))
    print(n_iterations)

    Config.seq_len = int(params["seq_len"])
    Config.batch_size = int(params["batch_size"])
    Config.learning_rate = params["learning_rate"]
    Config.l1 = int(params["l1"])
    Config.cnn_kernel_size = int(params["cnn_kernel_size"])
    Config.train_epochs = int(round(n_iterations))
    Config.loss = params["loss"]
    # BaseConfig.enc_in = int(params["PCA_n"])
    Config.D_kernel_size = int(params["D_kernel_size"])

    # gen_data(int(params["PCA_n"]), float(params["threshold"]), int(params["max_RUL"]))

    SCI = SCINetinitialization(Config)
    model = SCI.train("", False)
    mae, mse, err = SCI.predict("", False)

    return {"loss": err, "mae": mae, "mse": mse}, model
