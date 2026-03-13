import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from model import SegmentationModel

st.title("Medical Image Segmentation (Brain Tumor)")

# Upload MRI image
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["png","jpg","jpeg"])

# Load model once
@st.cache_resource
def load_model():
    model = SegmentationModel()
    model.load_state_dict(torch.load("brain_tumor_unet.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    st.write("Running segmentation...")

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output)

    # Convert mask to numpy
    mask = mask.squeeze().numpy()

    # Apply threshold
    mask = (mask > 0.5).astype(np.uint8)

    st.image(mask*255, caption="Predicted Tumor Mask", use_container_width=True)

else:
    st.info("Please upload a brain MRI image.")