import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import io

# 🎨 Page Config
st.set_page_config(
    page_title="Color Extractor",
    page_icon="🎨",
    layout="wide",
)

# 🌈 Custom Stylish CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rubik', sans-serif;
    background: linear-gradient(145deg, #f0f3f5 0%, #e9efff 100%);
}

h1, h2, h3, h4 {
    color: #202040;
}

.container-box {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(6px);
    border: 2px solid #dcd6f7;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 2rem;
}

.upload-box {
    background: linear-gradient(to bottom right, #c2ffd8, #465e91);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: inset 0 0 10px rgba(255,255,255,0.4);
    color: white;
}

.color-box {
    border-radius: 12px;
    padding: 1.2rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    color: white;
}

.download-box {
    background-color: #292f36;
    padding: 1rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.footer {
    text-align: center;
    font-size: 0.9rem;
    color: #5e6278;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# 💡 Functions
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(map(int, rgb))

def get_text_color(rgb):
    return 'black' if (rgb[0]*0.299 + rgb[1]*0.587 + rgb[2]*0.114) > 160 else 'white'

def extract_colors(image, n_colors=5):
    img_arr = np.array(image)
    pixels = img_arr.reshape(-1, 3)
    pixels = pixels[~np.all(pixels < 20, axis=1)]
    pixels = pixels[~np.all(pixels > 245, axis=1)]
    if pixels.size == 0:
        pixels = img_arr.reshape(-1, 3)
    model = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    model.fit(pixels)
    centers = model.cluster_centers_
    counter = Counter(model.labels_)
    total = sum(counter.values())
    results = [(centers[i], (counter[i] / total) * 100) for i in counter]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def create_palette_img(colors):
    fig, ax = plt.subplots(figsize=(6, 1))
    bar_colors = [tuple(c/255 for c in col[0]) for col in colors]
    ax.imshow([bar_colors])
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    plt.close()
    return buf

# 🎉 HEADER
st.markdown(f"""
<div class="container-box">
    <h1>🎨 Image Exctractor</h1>
    <h3><em>Where your image whispers in colors.</em></h3>
    <p>Upload an image and reveal its secret palette in a bold, styled presentation.</p>
</div>
""", unsafe_allow_html=True)

# 🖼️ IMAGE UPLOAD
uploaded_file = st.file_uploader("🔼 Drop or choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Your Uploaded Image", use_column_width=True)

    n_colors = st.slider("🎯 How many main colors?", 3, 10, 5)

    with st.spinner("🔍 Analyzing hues..."):
        palette = extract_colors(image, n_colors)

    st.markdown("## 🎨 Your Custom Palette")

    for i, (rgb, percent) in enumerate(palette):
        hex_val = rgb_to_hex(rgb)
        text_color = get_text_color(rgb)
        st.markdown(
            f'<div class="color-box" style="background-color:{hex_val};color:{text_color}">'
            f'Color {i+1}: {hex_val} | RGB{tuple(map(int, rgb))} | {percent:.1f}%'
            f'</div>', unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)

    with col1:
        img_buf = create_palette_img(palette)
        st.download_button(
            "⬇️ Download Palette Image",
            data=img_buf.getvalue(),
            file_name="palette_output.png",
            mime="image/png",
            use_container_width=True
        )

    with col2:
        text_data = "\n".join([f"Color {i+1}: {rgb_to_hex(rgb)} | RGB{tuple(map(int, rgb))} | {percent:.1f}%" for i, (rgb, percent) in enumerate(palette)])
        st.download_button(
            "📄 Save Palette Info (Text)",
            data=text_data,
            file_name="palette_data.txt",
            mime="text/plain",
            use_container_width=True
        )

# 👣 FOOTER
st.markdown("""
<div class="footer">
Created by Daniel Bintang W. Sitorus — NPM 140810230048
</div>
""", unsafe_allow_html=True)
