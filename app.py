import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super(SimpleGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.fc1 = nn.Linear(latent_dim + num_classes, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 28 * 28)

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.view(-1, 1, 28, 28)

# Load the model
@st.cache_resource
def load_model():
    model = SimpleGenerator().to("cpu")
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Generate digit images
def generate_images(generator, digit, num_samples=5):
    latent_dim = 64
    z = torch.randn(num_samples, latent_dim)
    labels = torch.full((num_samples,), digit, dtype=torch.long)
    with torch.no_grad():
        generated_imgs = generator(z, labels).cpu()
    return generated_imgs

# Streamlit UI
st.set_page_config(page_title="Handwritten Digit Generator", layout="centered")
st.title("🖋️ Handwritten Digit Generator")
st.markdown("Generate MNIST-style handwritten digits using a trained PyTorch model.")

digit = st.selectbox("Select a digit to generate", list(range(10)), index=2)
if st.button("Generate Images"):
    generator = load_model()
    images = generate_images(generator, digit)

    st.subheader(f"Generated images of digit: {digit}")
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i].squeeze(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
