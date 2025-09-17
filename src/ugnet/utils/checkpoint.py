from pathlib import Path
import torch

def load_state(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(Path(ckpt_path), map_location=map_location)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    state = {k.replace("module.","",1):v for k,v in state.items()}
    model.load_state_dict(state, strict=False)
    return model
