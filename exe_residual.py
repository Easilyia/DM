import argparse
import datetime
import json
import os

import numpy as np
import torch
import yaml

from dataset_residual import get_dataloader
from main_model import CSDI_Forecasting
from utils import evaluate, train


def parse_feature_columns(raw):
    if raw is None or raw.strip() == "":
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


parser = argparse.ArgumentParser(description="CSDI residual forecasting")
parser.add_argument("--config", type=str, default="base_residual.yaml")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--history_length", type=int, default=168)
parser.add_argument("--pred_length", type=int, default=24)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--valid_ratio", type=float, default=0.1)
parser.add_argument("--feature_columns", type=str, default="")
parser.add_argument("--time_column", type=str, default="")
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/residual_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)

feature_columns = parse_feature_columns(args.feature_columns)
time_column = args.time_column if args.time_column else None

(
    train_loader,
    valid_loader,
    test_loader,
    scaler,
    mean_scaler,
    target_dim,
    feature_names,
) = get_dataloader(
    data_path=args.data_path,
    device=args.device,
    batch_size=config["train"]["batch_size"],
    history_length=args.history_length,
    pred_length=args.pred_length,
    train_ratio=args.train_ratio,
    valid_ratio=args.valid_ratio,
    feature_columns=feature_columns,
    time_column=time_column,
)

config["data"] = {
    "data_path": args.data_path,
    "history_length": args.history_length,
    "pred_length": args.pred_length,
    "train_ratio": args.train_ratio,
    "valid_ratio": args.valid_ratio,
    "num_features": target_dim,
    "feature_columns": feature_names,
    "time_column": time_column,
}

with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

print(json.dumps(config, indent=4))

model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

model.target_dim = target_dim
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
