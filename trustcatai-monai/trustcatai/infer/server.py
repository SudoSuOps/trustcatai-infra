import os
import tempfile

import torch
from fastapi import FastAPI, UploadFile, File
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Spacing,
    Orientation,
    ScaleIntensityRange,
    EnsureType,
)

from ..trainer.train import build_model

app = FastAPI(title="trustcatai-infer")

model = None
preproc = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    Orientation(axcodes="RAS"),
    ScaleIntensityRange(a_min=-100, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    EnsureType(),
])

@app.on_event("startup")
def load_model():
    global model
    weights_path = os.environ.get("WEIGHTS", "/weights/best.ckpt")
    cfg_stub = type("Cfg", (object,), {})()
    model = build_model(cfg_stub)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        image = preproc(tmp.name).unsqueeze(0)
    os.unlink(tmp.name)

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        prob = torch.softmax(model(image), dim=1)
        score = (prob[:, 1] > 0.5).float().mean().item()
    return {"ok": True, "score": score}
