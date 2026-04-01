# utils/visualization.py
# Side-by-side comparison display for Streamlit.

import streamlit as st
from PIL import Image


def show_comparison(original: Image.Image, noisy: Image.Image, restored: Image.Image):
    """
    Display original, noisy, and restored images side by side in Streamlit.

    Args:
        original: The original clean input image.
        noisy: The image with synthetic noise applied.
        restored: The model's denoised output.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Original")
        st.image(original, use_container_width=True)

    with col2:
        st.markdown("### 🔊 Noisy")
        st.image(noisy, use_container_width=True)

    with col3:
        st.markdown("### ✨ Restored")
        st.image(restored, use_container_width=True)


def show_before_after(noisy: Image.Image, restored: Image.Image):
    """
    Show just noisy vs restored (when no separate original is available,
    e.g., when the user uploads an already-noisy document).
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔊 Input (Noisy)")
        st.image(noisy, use_container_width=True)

    with col2:
        st.markdown("### ✨ Restored")
        st.image(restored, use_container_width=True)
