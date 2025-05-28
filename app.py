import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import io
import base64
from collections import Counter

st.set_page_config(
    page_title="ColorSnap AI - Palette Hunter",
    page_icon="🌈",
    layout="wide",
)

# Custom CSS for unique visual style
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: #fbfaff;
    color: #333;
}
.header-box {
    background-color: #292f36;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    color: #fff;
    margin-bottom: 2rem;
    border: 4px solid #ffc300;
}
.header-box h1 {
    font-size: 2.8rem;
    margin: 0;
}
.header-box p {
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
.upload-card, .color-card, .footer-card {
    background-color: #fff;
    border: 2px solid #d1d1e9;
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
    box-shadow: 2px 4px 10px rgba(0,0,0,0.05);
}
.color-preview {
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    font-weight: 600;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.1);
}
.download-btn {
    background-color: #292f36;
    color: #fff;
    font-weight: bold;
    border-radius: 10px;
}
.footer-card h4 {
    color: #292f36;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(map(int, rgb))

def get_text_color(rgb):
    return 'black' if (rgb[0]*0.299 + rgb[1]*0.587 + rgb[2]*0.114) > 160 else 'white'

def extract_colors(image, n_colors=5):
    img_arr = np.array(image)
    flat_pixels = img_arr.reshape(-1, 3)
    flat_pixels = flat_pixels[~np.all(flat_pixels < 30, axis=1)]
    flat_pixels = flat_pixels[~np.all(flat_pixels > 225, axis=1)]
    if flat_pixels.size == 0:
        flat_pixels = img_arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=10).fit(flat_pixels)
    centers = kmeans.cluster_centers_
    count = Counter(kmeans.labels_)
    total = sum(count.values())
    palette = [(centers[i], (count[i] / total) * 100) for i in range(n_colors)]
    palette.sort(key=lambda x: x[1], reverse=True)
    return palette

def create_palette_img(colors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    rgb_colors = [c/255.0 for c, _ in colors]
    percentages = [p for _, p in colors]
    y = range(len(colors))
    ax1.barh(y, [1]*len(colors), color=rgb_colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f'Color {i+1}' for i in y])
    ax1.set_xlim(0, 1)
    ax1.set_title("Palette Bars")
    for i, (bar, (c, p)) in enumerate(zip(ax1.patches, colors)):
        text_color = 'black' if sum(c) > 400 else 'white'
        ax1.text(0.5, i, f"{rgb_to_hex(c)}\n{p:.1f}%", color=text_color,
                 ha='center', va='center', fontweight='bold')
    ax2.pie(percentages, labels=[rgb_to_hex(c) for c, _ in colors],
            colors=[c/255.0 for c, _ in colors], startangle=90,
            autopct='%1.1f%%')
    ax2.set_title("Palette Pie")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    return buf

# Header
st.markdown("""
<div class="header-box">
    <h1>🌈 ColorSnap AI</h1>
    <p>Find the soul of your image in 5 key colors. Fast. Smart. Beautiful.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for interaction
with st.sidebar:
    st.header("⚙️ Options")
    num_colors = st.slider("Number of colors to detect", 3, 10, 5)
    st.markdown("**Tips**: Try images with strong contrast for best results.")

# Main interface
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📁 Upload Image")
    image_file = st.file_uploader("Drag or select an image", type=["jpg", "jpeg", "png", "bmp"])
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.session_state.image = image
        st.markdown(f'<div class="upload-card">📐 Size: {image.size[0]} x {image.size[1]} px<br>📦 File: {image_file.name}</div>',
                    unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col_right:
    if image_file:
        st.markdown("### 🎯 Extracted Colors")
        with st.spinner("🎨 Scanning pixels..."):
            colors = extract_colors(image, num_colors)
            for i, (rgb, percent) in enumerate(colors):
                hex_code = rgb_to_hex(rgb)
                txt_color = get_text_color(rgb)
                st.markdown(f"""
                <div class="color-preview" style="background-color: {hex_code}; color: {txt_color};">
                    Color {i+1}: {hex_code}<br>RGB{tuple(map(int, rgb))} | {percent:.1f}%
                </div>
                """, unsafe_allow_html=True)

            palette_img = create_palette_img(colors)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "🖼️ Download Palette Image",
                    data=palette_img.getvalue(),
                    file_name=f"palette_{image_file.name.split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_dl2:
                text_data = "\n".join([
                    f"Color {i+1}: {rgb_to_hex(rgb)} | RGB{tuple(map(int, rgb))} | {percent:.1f}%"
                    for i, (rgb, percent) in enumerate(colors)
                ])
                st.download_button(
                    "📄 Download Palette Text",
                    data=text_data,
                    file_name=f"palette_{image_file.name.split('.')[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# Footer
st.markdown("""
<div class="footer-card">
    <h4>Created by Daniel Bintang W. Sitorus — 140810230048</h4>
</div>
""", unsafe_allow_html=True)
