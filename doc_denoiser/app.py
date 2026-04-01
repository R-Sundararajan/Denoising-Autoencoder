# app.py
# Streamlit UI for Document Image Denoising & Reconstruction.
# Supports both single images and multi-page PDFs.
#
# Run with:  streamlit run app.py

import io
import streamlit as st
from PIL import Image

from models import MODEL_REGISTRY
from inference.predict import load_model, denoise_image, check_weights_exist
from inference.image_utils import add_noise_to_pil, preprocess_image
from inference.pdf_utils import pdf_to_images, images_to_pdf
from utils.io_utils import ensure_directories
from utils.visualization import show_comparison, show_before_after

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Doc Denoiser",
    page_icon="✨",
    layout="wide",
)

# Create output directories on first run
ensure_directories()


# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# Model selector
model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(MODEL_REGISTRY.keys()),
    index=0,
    help="Choose the denoising model. U-Net is stronger but slower.",
)

# Noise strength slider (for adding synthetic noise to test the model)
add_noise = st.sidebar.checkbox("Add synthetic noise", value=True,
                                 help="Add noise to the input to test denoising quality.")
noise_strength = 0.0
if add_noise:
    noise_strength = st.sidebar.slider(
        "Noise Strength",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Higher = more noise. 0.05-0.15 is typical for documents.",
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tip:** Start with *Simple Autoencoder* for fast results. "
    "Switch to *U-Net* for higher quality."
)

# ── Main Content ─────────────────────────────────────────────────────────
st.title("📄 Document Denoiser")
st.markdown("Upload an image or PDF → add optional noise → denoise using deep learning.")
st.markdown("---")

# File uploader (images + PDFs)
uploaded_file = st.file_uploader(
    "Upload an image or PDF",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
    help="Supported formats: PNG, JPG, BMP, TIFF, PDF",
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    is_pdf = file_type == "pdf"

    # ── Check if model weights exist ─────────────────────────────────
    if not check_weights_exist(model_name):
        st.error(
            f"⚠️ **Weights not found for {model_name}.**\n\n"
            f"Please train the model first:\n\n"
            f"```\ncd doc_denoiser\n"
            f"python -m training.train --model \"{model_name}\" --data_dir data/raw\n```\n\n"
            f"Place the `.pth` file in `models/weights/`."
        )
        st.stop()

    # ── Load model (cached to avoid reloading every interaction) ─────
    @st.cache_resource
    def get_loaded_model(name):
        return load_model(name)

    try:
        model, device = get_loaded_model(model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # ── Process Image ────────────────────────────────────────────────
    if not is_pdf:
        # Load the uploaded image
        original_img = Image.open(uploaded_file).convert("L")

        # Optionally add noise
        if add_noise and noise_strength > 0:
            noisy_img = add_noise_to_pil(original_img, noise_strength)
        else:
            noisy_img = original_img.copy()

        # Run denoising
        with st.spinner("🔄 Denoising..."):
            restored_img = denoise_image(model, device, noisy_img)

        # Display comparison
        if add_noise and noise_strength > 0:
            show_comparison(original_img, noisy_img, restored_img)
        else:
            show_before_after(noisy_img, restored_img)

        # Download button
        st.markdown("---")
        buf = io.BytesIO()
        restored_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Restored Image",
            data=buf.getvalue(),
            file_name="restored_image.png",
            mime="image/png",
        )

    # ── Process PDF ──────────────────────────────────────────────────
    else:
        pdf_bytes = uploaded_file.read()
        pages = pdf_to_images(pdf_bytes)
        st.info(f"📑 PDF has **{len(pages)} page(s)**. Processing each page...")

        restored_pages = []
        progress_bar = st.progress(0)

        for i, page_img in enumerate(pages):
            # Convert to grayscale
            page_gray = page_img.convert("L")

            # Optionally add noise
            if add_noise and noise_strength > 0:
                noisy_page = add_noise_to_pil(page_gray, noise_strength)
            else:
                noisy_page = page_gray.copy()

            # Denoise
            restored_page = denoise_image(model, device, noisy_page)
            restored_pages.append(restored_page)

            # Update progress
            progress_bar.progress((i + 1) / len(pages))

        progress_bar.empty()
        st.success("✅ All pages denoised!")

        # Display each page's comparison
        for i, (page_img, restored_page) in enumerate(zip(pages, restored_pages)):
            st.markdown(f"### Page {i + 1}")
            page_gray = page_img.convert("L")
            if add_noise and noise_strength > 0:
                noisy_page = add_noise_to_pil(page_gray, noise_strength)
                show_comparison(page_gray, noisy_page, restored_page)
            else:
                show_before_after(page_gray, restored_page)
            st.markdown("---")

        # Download buttons for PDF output
        st.markdown("### 📥 Download Results")
        col1, col2 = st.columns(2)

        with col1:
            # Combined PDF download
            output_pdf_bytes = images_to_pdf(restored_pages)
            st.download_button(
                label="⬇️ Download Restored PDF",
                data=output_pdf_bytes,
                file_name="restored_document.pdf",
                mime="application/pdf",
            )

        with col2:
            # Individual page download (first page as sample)
            if restored_pages:
                buf = io.BytesIO()
                restored_pages[0].save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Page 1 as PNG",
                    data=buf.getvalue(),
                    file_name="restored_page_1.png",
                    mime="image/png",
                )

else:
    # No file uploaded — show instructions
    st.markdown(
        """
        ### 👋 Getting Started

        1. **Upload** an image or PDF using the uploader above.
        2. **Adjust** noise strength in the sidebar (optional).
        3. **Select** a denoising model.
        4. **View** the before/after comparison.
        5. **Download** the restored result.

        ---

        **Models available:**
        - **Simple Autoencoder** — Fast, good for light noise. *Start here.*
        - **U-Net** — Stronger, better for heavy noise and complex documents.

        **Need to train first?** See the README for training instructions.
        """
    )
