import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch
import timm
from torchvision import transforms
from PIL import Image
import streamlit as st

from config import IMAGE_MODEL_PATH

MODEL_PATH  = IMAGE_MODEL_PATH
NUM_CLASSES = 5
CLASS_NAMES = ["정상", "아토피", "건선", "여드름", "주사"]
IMG_SIZE    = 300

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = timm.create_model("tf_efficientnetv2_s", pretrained=False,
                                num_classes=NUM_CLASSES)
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, device

def predict(image: Image.Image):
    model, device = load_model()
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    results = sorted(
        [{"label": CLASS_NAMES[i], "prob": probs[i]} for i in range(NUM_CLASSES)],
        key=lambda x: x["prob"],
        reverse=True,
    )
    return results
