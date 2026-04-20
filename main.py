from io import BytesIO
from typing import Annotated
from urllib.request import urlopen

import timm
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import HttpUrl

app = FastAPI(title="NSFW Image Detection API")

MODEL_NAME = "hf_hub:Marqo/nsfw-image-detection-384"

model = timm.create_model(MODEL_NAME, pretrained=True)
model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
class_names = model.pretrained_cfg["label_names"]


def predict_image(img: Image.Image) -> dict:
    with torch.no_grad():
        output = model(transforms(img).unsqueeze(0)).softmax(dim=-1).cpu()[0]

    probs = output.tolist()
    pred_idx = int(output.argmax().item())

    return {
        "prob": float(probs[pred_idx]),
        "class_name": class_names[pred_idx],
    }


@app.get("/")
async def root():
    return {"message": "POST an image file or image_url to /predict"}


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile | None, File(None)] = None,
    image_url: Annotated[HttpUrl | None, Form(None)] = None,
):
    if file is None and image_url is None:
        raise HTTPException(
            status_code=400, detail="Provide either file or image_url"
        )

    if file is not None and image_url is not None:
        raise HTTPException(
            status_code=400, detail="Provide only one of file or image_url"
        )

    try:
        if file is not None:
            content = await file.read()
            img = Image.open(BytesIO(content)).convert("RGB")
        else:
            img = Image.open(urlopen(str(image_url))).convert("RGB")

        return predict_image(img)

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=False)
