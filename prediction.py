"""
prediction.py
"""

import torch


def predict(model, image_tensor, device):
    """
    Predict the most likely class.
    """

    model.eval()

    with torch.no_grad():

        image_tensor = image_tensor.unsqueeze(0).to(device)

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    return prediction.item(), confidence.item()


def predict_top5(model, image_tensor, device):
    """
    Return Top-5 predictions.
    """

    model.eval()

    with torch.no_grad():

        image_tensor = image_tensor.unsqueeze(0).to(device)

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        top_probs, top_ids = torch.topk(probabilities, k=5)

    return (
        top_ids.squeeze().cpu().numpy(),
        top_probs.squeeze().cpu().numpy()
    )