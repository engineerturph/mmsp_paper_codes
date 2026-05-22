import torch
import torch.nn as nn
import torch.nn.functional as F

class FallNet(nn.Module):
    def __init__(self):
        super(FallNet, self).__init__()
        
        # ==========================================
        # 1. LSTM KOLU (Zamansal Özellikler İçin)
        # ==========================================
        # Girdi boyutu 6 (ivme + jiroskop), 256 ünitelik LSTM katmanı
        self.lstm = nn.LSTM(input_size=6, hidden_size=256, batch_first=True)
        
        # LSTM kolu için Tam Bağlı (Dense) katmanlar
        self.lstm_fc1 = nn.Linear(256, 128)
        self.lstm_fc2 = nn.Linear(128, 64)
        self.lstm_fc3 = nn.Linear(64, 32)
        self.lstm_fc4 = nn.Linear(32, 8) # Çıkış: 8 Sınıf
        
        # ==========================================
        # 2. CNN KOLU (Uzamsal Özellikler İçin)
        # ==========================================
        # 128 filtreli, 3 boyutlu filtreye sahip 1D-CNN
        self.conv1d = nn.Conv1d(in_channels=6, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2) # Havuzlama
        
        # CNN kolu için Tam Bağlı (Dense) katmanlar
        # (Zaman adımı 200 -> Conv sonrası 198 -> Pool sonrası 99 kalır. 128 * 99 = 12672)
        self.cnn_fc1 = nn.Linear(128 * 99, 1024)
        self.cnn_fc2 = nn.Linear(1024, 512)
        self.cnn_fc3 = nn.Linear(512, 8) # Çıkış: 8 Sınıf

    def forward(self, x):
        # x'in beklenen boyutu: (Batch Sayısı, Zaman Adımı=200, Özellik=6)
        
        # --- LSTM Akışı ---
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :] 
        
        x_lstm = F.relu(self.lstm_fc1(lstm_out))
        x_lstm = F.relu(self.lstm_fc2(x_lstm))
        x_lstm = F.relu(self.lstm_fc3(x_lstm))
        lstm_probs = F.softmax(self.lstm_fc4(x_lstm), dim=1) 
        
        # --- CNN Akışı ---
        x_cnn = x.transpose(1, 2) 
        
        x_cnn = F.relu(self.conv1d(x_cnn))
        x_cnn = self.pool(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)
        
        x_cnn = F.relu(self.cnn_fc1(x_cnn))
        x_cnn = F.relu(self.cnn_fc2(x_cnn))
        # 8 Sınıflı Softmax Olasılıkları
        cnn_probs = F.softmax(self.cnn_fc3(x_cnn), dim=1) 
        
        # --- Ortak Çıktı (Ensemble) ---
        ensemble_probs = (lstm_probs + cnn_probs) / 2.0
        
        return ensemble_probs