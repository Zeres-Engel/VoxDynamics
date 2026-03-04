import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionDeepCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionDeepCNN, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv1d(1, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        
        # Layer 2
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout2 = nn.Dropout(0.2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout3 = nn.Dropout(0.2)
        
        # Layer 4
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout4 = nn.Dropout(0.2)
        
        # Layer 5
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout5 = nn.Dropout(0.2)
        
        # Flattening: 128 * 75 = 9600 (based on 2376 input length)
        self.fc1 = nn.Linear(9600, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (Batch, 1, 2376)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.dropout5(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

class EmotionCNNModel:
    def __init__(self, input_shape=None, num_classes=8):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionDeepCNN(num_classes=num_classes).to(self.device)

    def build(self):
        """Returns the PyTorch model."""
        return self.model

    def get_model(self):
        return self.model

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # We need to handle mapping if loading on different device
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
