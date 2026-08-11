import streamlit as st

st.set_page_config(
    page_title="Flower Classification System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)


import torch
import torch.nn.functional as F
import os
import random
import scipy.io

import numpy as np
from utils.gradcam import GradCAM, show_gradcam
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.efficientnet import create_efficientnet
from utils.flower_names import (
    ID_TO_NAME,
    NAME_TO_ID
)

from utils.flower_info import (
    FLOWER_INFO
)


# =====================================================
# Configuration
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "weights/best_efficientnet.pth"
DATASET_ROOT = "dataset/flowers-102"

FLOWERS_IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "jpg"
)

LABEL_FILE = os.path.join(
    DATASET_ROOT,
    "imagelabels.mat"
)


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="wide"
)


# =================================================
# Custom GUI Style
# =================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Prediction card */
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 20px;
    }

    /* Section title */
    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():

    model = create_efficientnet()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# =====================================================
# Image Transform
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =====================================================
# Prediction Function
# =====================================================

def predict_image(image):

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

    top_probs, top_ids = torch.topk(
        probabilities,
        5,
        dim=1
    )

    top_probs = top_probs[0].cpu().numpy()

    top_ids = top_ids[0].cpu().numpy()

    return top_ids, top_probs


# =================================================
# Find Flowers102 Example Images
# =================================================

def find_example_images(
    flower_id,
    max_images=3
):

    image_files = []

    # Flowers102 labels are 1-based
    target_label = flower_id + 1

    # Find image IDs belonging to this class
    matching_indices = [
        index
        for index, label in enumerate(
            flower_labels
        )
        if int(label) == target_label
    ]

    # Randomize examples
    random.shuffle(
        matching_indices
    )

    # Select requested number
    matching_indices = matching_indices[
        :max_images
    ]

    for index in matching_indices:

        image_number = index + 1

        filename = (
            f"image_{image_number:05d}.jpg"
        )

        image_path = os.path.join(
            FLOWERS_IMAGE_DIR,
            filename
        )

        if os.path.exists(image_path):

            image_files.append(
                image_path
            )

    return image_files
# =================================================
# Load Flowers102 Labels
# =================================================

@st.cache_data
def load_flowers102_labels():

    mat = scipy.io.loadmat(
        LABEL_FILE
    )

    labels = mat["labels"][0]

    return labels
flower_labels = load_flowers102_labels()

# =====================================================
# Title
# =====================================================

st.markdown(
    """
    <div class="main-title">
        🌸 Flower Classification System
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    """
    Upload a flower image to classify it into one of the
    102 flower categories from the Flowers102 dataset.

    The system provides the predicted flower name,
    confidence score, Top-5 predictions, and flower information.
    """
)


st.markdown(
    """
    <div class="subtitle">
        Deep Learning Based Flower Recognition using EfficientNet-B0
    </div>
    """,
    unsafe_allow_html=True
)
st.write(
    "AI-based flower recognition using EfficientNet."
)

# =================================================
# Sidebar
# =================================================

with st.sidebar:

    st.header("🌸 Project Information")

    st.write(
        "This application uses a deep learning model "
        "to classify flowers into 102 categories."
    )

    st.divider()

    st.subheader("Model")

    st.write(
        "EfficientNet-B0"
    )

    st.subheader("Dataset")

    st.write(
        "Oxford Flowers 102"
    )

    st.subheader("Classes")

    st.write(
        "102 flower categories"
    )

    st.divider()

    st.caption(
        "Flower Classification Project"
    )
# =================================================
# Upload Flower Image
# =================================================

st.markdown(
    '<div class="section-title">📷 Upload Flower Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png"],
    help="Upload a flower image for classification."
)

if uploaded_file is not None:

    # Load uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(
        [1, 1.5]
    )

    # ---------------------------------------------
    # Left: Uploaded Image
    # ---------------------------------------------

    with col1:

        st.subheader(
            "📷 Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    # ---------------------------------------------
    # Right: Prediction
    # ---------------------------------------------

    with col2:

        st.subheader(
            "🎯 Prediction Result"
        )

        # -----------------------------------------
        # Transform image
        # -----------------------------------------

        image_tensor = transform(
            image
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # -----------------------------------------
        # Model prediction
        # -----------------------------------------

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            predicted_class = (
                probabilities.argmax(
                    dim=1
                ).item()
            )

            predicted_probability = (
                probabilities[0, predicted_class]
                .item()
            )

        # -----------------------------------------
        # Flower name
        # -----------------------------------------

        flower_name = ID_TO_NAME[
            predicted_class
        ]

        st.success(
            f"🌸 {flower_name}"
        )

        st.metric(
            "Confidence",
            f"{predicted_probability:.2%}"
        )
    # =================================================
    # Top-5 Predictions
    # =================================================

    st.subheader(
        "🏆 Top-5 Predictions"
    )

    # Get Top-5 classes and probabilities
    top_probs, top_ids = torch.topk(
        probabilities[0],
        k=5
    )

    for rank, (class_id, probability) in enumerate(
            zip(top_ids, top_probs),
            start=1
    ):
        class_id = int(
            class_id.item()
        )

        probability = float(
            probability.item()
        )

        flower_name = ID_TO_NAME[
            class_id
        ]

        col1, col2, col3 = st.columns(
            [0.15, 0.55, 0.30]
        )

        with col1:
            st.write(
                f"**#{rank}**"
            )

        with col2:
            st.write(
                f"**{flower_name}**"
            )

        with col3:
            st.write(
                f"{probability:.2%}"
            )

        st.progress(
            probability
        )

# =================================================
# Flower Category Search
# =================================================

st.divider()

st.subheader(
    "🔍 Flower Category Search"
)

st.write(
    "Search for a flower name to view its category ID "
    "and example images from the dataset."
)

search_name = st.text_input(
    "Enter a flower name",
    placeholder="Example: rose, sunflower, tulip..."
)
if search_name:

    query = search_name.strip().lower()

    # Exact match
    if query in NAME_TO_ID:

        flower_id = NAME_TO_ID[query]
        flower_name = ID_TO_NAME[flower_id]

        # =================================================
        # Example Images
        # =================================================

        example_images = find_example_images(
            flower_id=flower_id,
            max_images=3
        )

        if example_images:

            st.subheader(
                "🖼️ Example Images"
            )

            image_columns = st.columns(
                len(example_images)
            )

            for column, image_path in zip(
                    image_columns,
                    example_images
            ):
                with column:
                    example_image = Image.open(
                        image_path
                    ).convert("RGB")

                    st.image(
                        example_image,
                        use_container_width=True
                    )

        else:

            st.info(
                "No example images were found "
                "for this category."
            )

        # =================================================
        # Flower Information
        # =================================================

        flower_key = flower_name.lower()

        if flower_key in FLOWER_INFO:
            st.subheader(
                "📖 Flower Information"
            )

            st.write(
                FLOWER_INFO[flower_key]
            )

        st.success(
            f"Found: {flower_name}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Category ID",
                flower_id
            )

        with col2:

            st.metric(
                "Flower Name",
                flower_name
            )

    else:

        # Partial search
        matched_flowers = [
            (flower_id, flower_name)
            for flower_id, flower_name in ID_TO_NAME.items()
            if query in flower_name.lower()
        ]

        if matched_flowers:

            st.info(
                f"Found {len(matched_flowers)} matching categories."
            )

            for flower_id, flower_name in matched_flowers:

                st.write(
                    f"**Class {flower_id}: {flower_name}**"
                )

        else:

            st.warning(
                "No matching flower was found."
            )

# =================================================
# Browse All Flower Classes
# =================================================

with st.expander(
    "📚 Browse All 102 Flower Categories"
):

    for flower_id, flower_name in ID_TO_NAME.items():

        st.write(
            f"**{flower_id}** — {flower_name}"
        )



# =================================================
# Grad-CAM Visualization
# =================================================

def create_gradcam_overlay(
    original_image,
    cam_map,
    alpha=0.45
):

    original_image = np.array(
        original_image
    )

    # Normalize CAM
    cam_map = np.maximum(
        cam_map,
        0
    )

    if cam_map.max() > 0:
        cam_map = (
            cam_map /
            cam_map.max()
        )

    # Resize CAM to original image size
    from PIL import Image as PILImage

    cam_image = PILImage.fromarray(
        (cam_map * 255).astype(np.uint8)
    )

    cam_image = cam_image.resize(
        (
            original_image.shape[1],
            original_image.shape[0]
        ),
        PILImage.Resampling.BILINEAR
    )

    cam_map_resized = (
        np.array(cam_image) / 255.0
    )

    # Apply colormap
    heatmap = cm.jet(
        cam_map_resized
    )[:, :, :3]

    # Convert original image to float
    original_float = (
        original_image / 255.0
    )

    # Overlay
    overlay = (
        (1 - alpha) * original_float
        +
        alpha * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return heatmap, overlay


st.markdown(
    """
    <div class="footer">
        🌸 Flower Classification Project |
        PyTorch + EfficientNet-B0 |
        Oxford Flowers 102
    </div>
    """,
    unsafe_allow_html=True
)