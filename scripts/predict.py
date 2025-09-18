import argparse,glob,torch
from PIL import Image
import torchvision.transforms as T
from ugnet.models.umcc import UMCC
from ugnet.models.mag import MAG
from ugnet.utils.checkpoint import load_state

def load_img(p,size=224):
    return T.Compose([T.Resize((size,size)),T.ToTensor()])(Image.open(p).convert("RGB")).unsqueeze(0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch",choices=["umcc","mag"],default="umcc")
    ap.add_argument("--backbone",default="tf_efficientnetv2_s")
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--modality",choices=["hand","face"],default="face")
    ap.add_argument("--glob",required=True)
    args=ap.parse_args()

    model=UMCC(args.backbone) if args.arch=="umcc" else MAG(args.backbone)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=load_state(model,args.ckpt,map_location=device).to(device).eval()

    for p in sorted(glob.glob(args.glob)):
        x=load_img(p).to(device)
        prob=model.infer(x,args.modality).item() if args.arch=="umcc" else model.infer(x).item()
        print(f"{p}\tprob_male={prob:.3f}\tpred={'male' if prob>=0.5 else 'female'}")

if __name__=="__main__": main()
