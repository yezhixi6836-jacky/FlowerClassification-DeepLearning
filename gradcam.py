import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None

        # Forward hook
        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):

        self.activations = output

    def generate(self, image_tensor, class_idx=None):

        self.model.eval()

        # Make sure gradients are enabled
        image_tensor = image_tensor.clone().detach()
        image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)

        # Predicted class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Clear gradients
        self.model.zero_grad(set_to_none=True)

        # Target score
        score = output[:, class_idx].sum()

        # Calculate gradients of target score
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=self.activations,
            retain_graph=True,
            create_graph=False,
            allow_unused=False
        )[0]

        # Get activation
        activations = self.activations

        # Remove batch dimension
        activations = activations[0]
        gradients = gradients[0]

        # Global Average Pooling
        weights = gradients.mean(
            dim=(1, 2)
        )

        # Weighted feature maps
        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )

        for i in range(activations.shape[0]):

            cam += (
                weights[i]
                * activations[i]
            )

        # ReLU
        cam = torch.relu(cam)

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return cam.detach().cpu().numpy()

    def remove_hooks(self):

        self.forward_handle.remove()


def create_overlay(
    image,
    cam,
    alpha=0.4
):

    # Convert image to uint8
    if image.dtype != np.uint8:

        image = (
            image * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

    # Resize CAM
    cam = cv2.resize(
        cam,
        (
            image.shape[1],
            image.shape[0]
        )
    )

    # Convert CAM to 0-255
    heatmap = np.uint8(
        255 * cam
    )

    # Apply color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Convert BGR -> RGB
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return heatmap, overlay


def show_gradcam(
    image,
    cam,
    title="Grad-CAM"
):

    heatmap, overlay = create_overlay(
        image,
        cam
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)

    plt.imshow(image)

    plt.title(
        "Original Image"
    )

    plt.axis("off")

    plt.subplot(1, 3, 2)

    plt.imshow(heatmap)

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis("off")

    plt.subplot(1, 3, 3)

    plt.imshow(overlay)

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.show()

    return heatmap, overlay