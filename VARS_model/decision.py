import torch
from torch import nn

# -----------------------------------------------------------
# 1.  DÖRT FEATURE HEAD'i  (+ Sigmoid → 0-1 arası değer)
# -----------------------------------------------------------
class FourHeads(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        def _head():  # ortak şablon
            return nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, feat_dim),
                nn.Linear(feat_dim, 1),
                nn.Sigmoid()
            )
        self.action_cls  = _head()      # Action Class
        self.multi_foul  = _head()      # Multiple Fouls
        self.try_play    = _head()      # Try to Play
        self.touch_ball  = _head()      # Touch Ball

    def forward(self, x):
        return {
            "action_cls":  self.action_cls(x).squeeze(-1),
            "multi_foul":  self.multi_foul(x).squeeze(-1),
            "try_play":    self.try_play(x).squeeze(-1),
            "touch_ball":  self.touch_ball(x).squeeze(-1),
        }

# -----------------------------------------------------------
# 2.  KARAR KURALI  (formül + eşikler)
# -----------------------------------------------------------
@torch.no_grad()
def decide_from_feats(feats):
    """
    feats: dict – anahtarlar:
        action_cls, multi_foul, try_play, touch_ball   (B,)
    dönen: decision  (LongTensor, B,)
    """
    O = 0.6 * feats["action_cls"] + 0.4 * feats["multi_foul"]
    M = 0.7 * feats["try_play"]   + 0.3 * feats["touch_ball"]
    S = O - M  # (B,)
    # Eşikleme
    decision = torch.zeros_like(S, dtype=torch.long)
    decision[(S >= 0.30) & (S < 0.45)] = 1
    decision[(S >= 0.45) & (S < 0.60)] = 2
    decision[S >= 0.60]                = 3
    return decision

# Decision sınıfları
DECISION_CLASSES = {
    0: "No Foul",
    1: "Foul - No Card",
    2: "Foul - Yellow Card",
    3: "Foul - Red Card"
} 