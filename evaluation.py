"""
evaluation.py
"""

import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


@torch.no_grad()
def evaluate_model(model, loader, device):

    model.eval()

    predictions = []
    labels_list = []

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1).cpu()

        predictions.extend(preds.numpy())

        labels_list.extend(labels.numpy())

    accuracy = accuracy_score(labels_list, predictions)

    report = classification_report(
        labels_list,
        predictions,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        labels_list,
        predictions
    )

    return accuracy, report, cm