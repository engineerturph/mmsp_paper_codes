import torch
import torch.nn as nn
import torch.nn.functional as F

CNN_FILTERS = 128
GRU_HIDDEN = 128
DROPOUT_RATE = 0.3
NUM_CLASSES = 3


class CNNFeatureExtractor(nn.Module):
    """FallAllD-style multi-scale 1D CNN feature extractor."""

    def __init__(self, in_channels, filters=CNN_FILTERS):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(filters, filters, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(filters)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(filters, filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(filters)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class CNN_GRU(nn.Module):
    def __init__(
        self,
        in_channels=9,
        num_classes=NUM_CLASSES,
        cnn_filters=CNN_FILTERS,
        gru_hidden=GRU_HIDDEN,
        num_layers=2,
        dropout_rate=DROPOUT_RATE,
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels, cnn_filters)
        self.gru = nn.GRU(
            input_size=cnn_filters,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        self.gru_norm = nn.LayerNorm(gru_hidden)
        self.fc1 = nn.Linear(gru_hidden, 64)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x).transpose(1, 2)
        gru_out, _ = self.gru(cnn_out)
        last_hidden = self.gru_norm(gru_out[:, -1, :])
        hidden = F.relu(self.fc1(last_hidden))
        hidden = self.dropout_fc1(hidden)
        return self.fc2(hidden)
