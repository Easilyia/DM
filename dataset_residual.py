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
    dataset = ResidualForecastingDataset(
        data_path=data_path,
        mode="train",
        history_length=history_length,
        pred_length=pred_length,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        feature_columns=feature_columns,
        time_column=time_column,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

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
