import torch, torch.nn as nn, torch.nn.functional as F
import timm

class MAG(nn.Module):
    """Modality-Aware Gated network"""
    def __init__(self, backbone="tf_efficientnetv2_s", gating="soft", lambda_mod=0.3):
        super().__init__()
        self.gating, self.lambda_mod = gating, lambda_mod
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.gate = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,2))
        self.hand_head, self.face_head = nn.Linear(d,1), nn.Linear(d,1)

    def forward(self, x, y=None, m_true=None):
        f = self.backbone(x)
        g_logits = self.gate(f)
        w = F.softmax(g_logits,1)
        lh, lf = self.hand_head(f).squeeze(1), self.face_head(f).squeeze(1)
        logits = w[:,0]*lh + w[:,1]*lf if self.gating=="soft" else torch.where(w.argmax(1)==0, lh, lf)
        if y is None: return logits
        loss_gender = F.binary_cross_entropy_with_logits(logits, y.float())
        loss_mod = F.cross_entropy(g_logits, m_true.argmax(1))
        return logits, loss_gender + self.lambda_mod*loss_mod

    @torch.no_grad()
    def infer(self, img):
        self.eval()
        return torch.sigmoid(self.forward(img))
