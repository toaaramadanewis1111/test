import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# import your UNet ONLY (copy class here OR import it)
from main import UNet   # if same folder

# ------------------
# LOAD MODEL
# ------------------
@st.cache_resource
def load_model():
    model = UNet()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------
# IMAGE TRANSFORM
# ------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# ------------------
# UI
# ------------------
st.title("🧠 Face Inpainting AI")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original")

    x = transform(img).unsqueeze(0)

    # fake mask (same logic as training)
    mask = torch.ones_like(x)
    h,w = x.shape[2], x.shape[3]

    mh, mw = h//4, w//4
    y = torch.randint(0,h-mh,(1,)).item()
    x_ = torch.randint(0,w-mw,(1,)).item()

    mask[:,:,y:y+mh,x_:x_+mw] = 0

    x_masked = x * mask

    with torch.no_grad():
        output = model(x_masked)

    # convert back
    output = output[0].permute(1,2,0).numpy()
    output = (output + 1) / 2

    st.image(output, caption="Reconstructed")