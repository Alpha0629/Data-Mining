import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    """CNN model with convolutional encoder and linear classifier.
    
    Designed for images (64x64).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super(CNN, self).__init__()
        
        # Encoder: 卷积特征提取层
        # 输入: 3 x 64 x 64
        self.encoder = nn.Sequential(
            # 第一个卷积块: 3 -> 32
            # 64x64 -> 64x64 (padding=1保持尺寸) -> 32x32 (pool)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
            
            # 第二个卷积块: 32 -> 64
            # 32x32 -> 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            
            # 第三个卷积块: 64 -> 128
            # 16x16 -> 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
        )
        
        # Classifier: 线性分类层
        # 输入尺寸: 128 * 8 * 8 = 8192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)



class CNN_EncoderOnly(nn.Module):
    """
    用于对比学习阶段的 Encoder + Projection Head
    """
    def __init__(self, dropout=0.1, projection_dim=128):
        super().__init__()
        from model.CNN import CNN

        self.encoder = nn.Sequential(
            CNN(num_classes=2, dropout=dropout).encoder,
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(feat, dim=1)
        return feat
