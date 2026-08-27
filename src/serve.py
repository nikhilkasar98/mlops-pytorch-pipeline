import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from .model import get_model


app = FastAPI(
    title="PyTorch CIFAR-10 Model Server",
    version="1.0.0",
)


# Configuration
CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT",
    "checkpoints/classifier_v1.pt",
)

ARCHITECTURE = os.getenv(
    "MODEL_ARCHITECTURE",
    "resnet18",
)

NUM_CLASSES = int(
    os.getenv(
        "MODEL_NUM_CLASSES",
        "10",
    )
)


# CIFAR-10 normalization
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


model = None
model_loaded = False


def load_model():
    global model
    global model_loaded

    checkpoint_path = Path(CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    model = get_model(
        architecture=ARCHITECTURE,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    model_loaded = True


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    if model_loaded:
        return {
            "status": "healthy",
            "model_loaded": True,
        }

    raise HTTPException(
        status_code=503,
        detail="Model is not loaded",
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if not image.content_type or not image.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image",
        )

    try:
        image_bytes = await image.read()

        input_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read uploaded image",
        ) from exc

    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    probabilities_list = probabilities.tolist()

    class_probabilities = {
        CIFAR10_CLASSES[i]: round(
            probabilities_list[i],
            6,
        )
        for i in range(NUM_CLASSES)
    }

    return {
        "predicted_class": CIFAR10_CLASSES[predicted_index],
        "predicted_class_index": predicted_index,
        "probabilities": class_probabilities,
    }