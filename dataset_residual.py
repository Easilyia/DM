import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ResidualForecastingDataset(Dataset):
    def __init__(
        self,
        data_path,
        mode="train",
        history_length=168,
        pred_length=24,
        train_ratio=0.7,
        valid_ratio=0.1,
        feature_columns=None,
        time_column=None,
    ):
        self.history_length = history_length
        self.pred_length = pred_length
        self.seq_length = history_length + pred_length

        if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
            raise ValueError("train_ratio and valid_ratio must satisfy 0 < train_ratio, valid_ratio and train_ratio + valid_ratio < 1")

        df = pd.read_csv(data_path)
        if time_column is not None and time_column in df.columns:
            df = df.drop(columns=[time_column])

        if feature_columns is not None and len(feature_columns) > 0:
            missing_cols = [c for c in feature_columns if c not in df.columns]
            if missing_cols:
                raise ValueError(f"feature columns not found: {missing_cols}")
            df = df[feature_columns]
        else:
            df = df.select_dtypes(include=[np.number])

        if df.shape[1] == 0:
            raise ValueError("no numeric feature columns found in data")

        data = df.values.astype(np.float32)
        total_length = len(data)
        if total_length < self.seq_length + 2:
            raise ValueError(
                f"time length is too short ({total_length}); it must be at least {self.seq_length + 2}"
            )

        train_end = int(total_length * train_ratio)
        valid_end = int(total_length * (train_ratio + valid_ratio))

        if train_end <= self.seq_length:
            raise ValueError("train split is too short for the configured history_length and pred_length")
        if valid_end - train_end <= self.pred_length:
            raise ValueError("valid split is too short; increase data length or adjust split ratios")
        if total_length - valid_end <= self.pred_length:
            raise ValueError("test split is too short; increase data length or adjust split ratios")

        train_mean = data[:train_end].mean(axis=0)
        train_std = data[:train_end].std(axis=0)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)

        self.mean_data = train_mean.astype(np.float32)
        self.std_data = train_std.astype(np.float32)

        self.main_data = ((data - self.mean_data) / self.std_data).astype(np.float32)
        self.mask_data = np.ones_like(self.main_data, dtype=np.float32)
        self.feature_names = list(df.columns)

        if mode == "train":
            start = 0
            end = train_end - self.seq_length + 1
            step = 1
        elif mode == "valid":
            start = max(0, train_end - self.history_length)
            end = valid_end - self.seq_length + 1
            step = self.pred_length
        elif mode == "test":
            start = max(0, valid_end - self.history_length)
            end = total_length - self.seq_length + 1
            step = self.pred_length
        else:
            raise ValueError("mode must be one of train/valid/test")

        if end <= start:
            raise ValueError(
                f"no available windows for mode={mode}; start={start}, end={end}, seq_length={self.seq_length}"
            )

        self.use_index = np.arange(start, end, step)

    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index : index + self.seq_length].copy()
        target_mask[-self.pred_length :] = 0.0

        return {
            "observed_data": self.main_data[index : index + self.seq_length],
            "observed_mask": self.mask_data[index : index + self.seq_length],
            "gt_mask": target_mask,
            "timepoints": np.arange(self.seq_length, dtype=np.float32),
            "feature_id": np.arange(self.main_data.shape[1], dtype=np.float32),
        }

    def __len__(self):
        return len(self.use_index)


class ResidualSplitNpyDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode="train",
        pred_length=24,
        train_mean=None,
        train_std=None,
        feature_names=None,
    ):
        file_map = {
            "train": "train_res.npy",
            "valid": "val_res.npy",
            "test": "test_res.npy",
        }
        if mode not in file_map:
            raise ValueError("mode must be one of train/valid/test")

        data_path = os.path.join(data_dir, file_map[mode])
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"required file not found: {data_path}")

        data = np.load(data_path).astype(np.float32)
        if data.ndim != 3:
            raise ValueError(
                f"{data_path} must be 3D with shape (N, L, K), but got {data.shape}"
            )

        n_sample, seq_length, n_feature = data.shape
        if n_sample <= 0:
            raise ValueError(f"{data_path} is empty")
        if pred_length <= 0 or pred_length >= seq_length:
            raise ValueError(
                f"pred_length must satisfy 0 < pred_length < seq_length ({seq_length}), got {pred_length}"
            )

        if mode == "train":
            self.mean_data = data.reshape(-1, n_feature).mean(axis=0).astype(np.float32)
            self.std_data = data.reshape(-1, n_feature).std(axis=0).astype(np.float32)
            self.std_data = np.where(self.std_data < 1e-6, 1.0, self.std_data)
        else:
            if train_mean is None or train_std is None:
                raise ValueError("train_mean and train_std are required for valid/test")
            self.mean_data = train_mean.astype(np.float32)
            self.std_data = np.where(train_std.astype(np.float32) < 1e-6, 1.0, train_std.astype(np.float32))

        self.main_data = ((data - self.mean_data) / self.std_data).astype(np.float32)
        self.mask_data = np.ones_like(self.main_data, dtype=np.float32)
        self.pred_length = pred_length
        self.seq_length = seq_length

        if feature_names is not None:
            if len(feature_names) != n_feature:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) does not match feature dim ({n_feature})"
                )
            self.feature_names = feature_names
        elif n_feature == 3:
            self.feature_names = ["wind", "solar", "load"]
        else:
            self.feature_names = [f"feature_{i}" for i in range(n_feature)]

    def __getitem__(self, orgindex):
        target_mask = self.mask_data[orgindex].copy()
        target_mask[-self.pred_length :] = 0.0

        return {
            "observed_data": self.main_data[orgindex],
            "observed_mask": self.mask_data[orgindex],
            "gt_mask": target_mask,
            "timepoints": np.arange(self.seq_length, dtype=np.float32),
            "feature_id": np.arange(self.main_data.shape[2], dtype=np.float32),
        }

    def __len__(self):
        return len(self.main_data)


def get_dataloader(
    data_path,
    device,
    batch_size=8,
    history_length=168,
    pred_length=24,
    train_ratio=0.7,
    valid_ratio=0.1,
    feature_columns=None,
    time_column=None,
):
    if os.path.isdir(data_path):
        train_file = os.path.join(data_path, "train_res.npy")
        valid_file = os.path.join(data_path, "val_res.npy")
        test_file = os.path.join(data_path, "test_res.npy")
        if not (os.path.exists(train_file) and os.path.exists(valid_file) and os.path.exists(test_file)):
            raise FileNotFoundError(
                f"directory mode requires train_res.npy/val_res.npy/test_res.npy under {data_path}"
            )

        train_dataset = ResidualSplitNpyDataset(
            data_dir=data_path,
            mode="train",
            pred_length=pred_length,
        )
        valid_dataset = ResidualSplitNpyDataset(
            data_dir=data_path,
            mode="valid",
            pred_length=pred_length,
            train_mean=train_dataset.mean_data,
            train_std=train_dataset.std_data,
            feature_names=train_dataset.feature_names,
        )
        test_dataset = ResidualSplitNpyDataset(
            data_dir=data_path,
            mode="test",
            pred_length=pred_length,
            train_mean=train_dataset.mean_data,
            train_std=train_dataset.std_data,
            feature_names=train_dataset.feature_names,
        )
        dataset = train_dataset
    else:
        train_dataset = ResidualForecastingDataset(
            data_path=data_path,
            mode="train",
            history_length=history_length,
            pred_length=pred_length,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            feature_columns=feature_columns,
            time_column=time_column,
        )
        valid_dataset = ResidualForecastingDataset(
            data_path=data_path,
            mode="valid",
            history_length=history_length,
            pred_length=pred_length,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            feature_columns=feature_columns,
            time_column=time_column,
        )
        test_dataset = ResidualForecastingDataset(
            data_path=data_path,
            mode="test",
            history_length=history_length,
            pred_length=pred_length,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            feature_columns=feature_columns,
            time_column=time_column,
        )
        dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return (
        train_loader,
        valid_loader,
        test_loader,
        scaler,
        mean_scaler,
        len(dataset.feature_names),
        dataset.feature_names,
    )
