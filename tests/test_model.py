import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.model import get_model
from src import serve


def test_model_creation():
    """Verify that the ResNet-18 model can be created."""
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    assert model is not None


def test_model_output_shape():
    """Verify that the model accepts CIFAR-10 images and returns 10 classes."""
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    model.eval()

    # CIFAR-10 image: batch_size=1, channels=3, height=32, width=32
    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (1, 10)


def test_model_checkpoint():
    """Verify that the trained checkpoint can be loaded."""
    checkpoint_path = "checkpoints/classifier_v1.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (1, 10)


def test_model_produces_finite_output():
    """Verify that model predictions contain valid finite values."""
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    model.eval()

    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert torch.isfinite(outputs).all()


from fastapi.testclient import TestClient


def test_health_endpoint():
    """Verify that the health endpoint reports a loaded model."""
    serve.model_loaded = True

    client = TestClient(serve.app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    """Verify that the prediction endpoint returns class probabilities."""
    serve.model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    serve.model.eval()
    serve.model_loaded = True

    client = TestClient(serve.app)

    from PIL import Image
    import io

    image = Image.new(
        "RGB",
        (32, 32),
        color=(128, 128, 128),
    )

    image_buffer = io.BytesIO()

    image.save(
        image_buffer,
        format="PNG",
    )

    image_buffer.seek(0)

    response = client.post(
        "/predict",
        files={
            "image": (
                "test.png",
                image_buffer,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_class" in data
    assert "predicted_class_index" in data
    assert "probabilities" in data

    assert len(data["probabilities"]) == 10