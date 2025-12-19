import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    """
    ResNet50 model with encoder-classifier structure.
    
    Encoder: ResNet50 backbone (conv1 -> avgpool)
    Classifier: Final fully connected layer
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super().__init__()

        # 加载 ResNet50 预训练模型
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        # 修改第一层：适配 64x64 小图
        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        resnet.maxpool = nn.Identity()

        # Encoder: ResNet50 backbone (从 conv1 到 avgpool，不包括 fc)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten()  # 将特征图展平
        )

        # Classifier: 最后的全连接层
        in_dim = resnet.fc.in_features  # ResNet50 的 fc 输入维度是 2048
        self.classifier = nn.Linear(in_dim, num_classes)

        # 冻结 encoder（只训练 classifier）
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classifier 层要始终可训练
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [batch, 3, H, W]
        
        Returns:
            logits: 分类logits [batch, num_classes]
        """
        features = self.encoder(x)  # [batch, 2048]
        logits = self.classifier(features)  # [batch, num_classes]
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征表示（用于特征提取、可视化等）
        
        Args:
            x: 输入图像 [batch, 3, H, W]
        
        Returns:
            features: 特征向量 [batch, 2048]
        """
        return self.encoder(x)

