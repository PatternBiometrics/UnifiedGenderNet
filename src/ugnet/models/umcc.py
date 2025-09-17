import torch, torch.nn as nn
import timm

class UMCC(nn.Module):
    """Unified Modality-Conditioned Classifier"""
    def __init__(self, backbone="efficientnetv2_s", flag_on=True, drop_rate=0.2):
        super().__init__()
        self.flag_on = flag_on
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0,
                                          global_pool="avg", drop_rate=drop_rate)
        feat = self.backbone.num_features
        self.classifier = nn.Linear(feat + (2 if flag_on else 0), 1)

    def forward(self, x, flag):
        feats = self.backbone(x)
        if self.flag_on:
            feats = torch.cat([feats, flag], dim=1)
        return self.classifier(feats).squeeze(1)

    @torch.no_grad()
    def infer(self, img, modality: str):
        self.eval()
        flag = torch.tensor([[1,0] if modality=="hand" else [0,1]], device=img.device, dtype=img.dtype)
        flag = flag.repeat(img.size(0),1)
        return torch.sigmoid(self.forward(img, flag))
