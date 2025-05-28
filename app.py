import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import io
import base64
from collections import Counter

st.set_page_config(
    page_title="🎨 Color Palette Extractor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    .main .block-container {
        background: transparent;
    }
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        color: #ffffff !important;
    }
    .main-header p {
        font-size: 1.2rem;
        margin: 1rem 0 0;
        color: #e0e7ff;
    }
    .color-box {
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: 0.3s;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .color-box:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        text-align: center;
        color: #f1f5f9;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .download-link {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
    }
    .footer-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        text-align: center;
        color: #f1f5f9;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .footer-container h3 {
        background: linear-gradient(45deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_text_color(rgb):
    brightness = (rgb[0]*299 + rgb[1]*587 + rgb[2]*114) / 1000
    return 'black' if brightness > 128 else 'white'

def extract_colors(image, n_colors=5):
    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)
    pixels = pixels[~np.all(pixels < 30, axis=1)]
    pixels = pixels[~np.all(pixels > 225, axis=1)]
    if len(pixels) == 0:
        pixels = image_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    label_counts = Counter(labels)
    total_pixels = len(labels)
    sorted_colors = [(colors[i], (label_counts[i] / total_pixels) * 100) for i in range(n_colors)]
    sorted_colors.sort(key=lambda x: x[1], reverse=True)
    return sorted_colors

def create_palette_image(colors_with_percentages):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = [color for color, _ in colors_with_percentages]
    percentages = [percentage for _, percentage in colors_with_percentages]
    y_pos = range(len(colors))
    bars = ax1.barh(y_pos, [1]*len(colors), color=[tuple(c/255.0) for c in colors])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f'Color {i+1}' for i in range(len(colors))])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Color Palette')
    ax1.set_title('Extracted Color Palette', fontsize=16, fontweight='bold')
    for i, (bar, (color, percentage)) in enumerate(zip(bars, colors_with_percentages)):
        hex_color = rgb_to_hex(color)
        text_color = 'white' if sum(color) < 400 else 'black'
        ax1.text(0.5, i, f'{hex_color}\n{percentage:.1f}%', ha='center', va='center', fontweight='bold', color=text_color)
    ax2.pie(percentages, colors=[tuple(c/255.0) for c in colors],
            labels=[rgb_to_hex(color) for color, _ in colors_with_percentages],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Color Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    return img_buffer

# Header
st.markdown("""
<div class="main-header">
    <h1>🎨 Color Palette Extractor</h1>
    <p>Extract dominant colors from your images with AI-powered color analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    n_colors = st.slider("Number of colors to extract", 3, 8, 5)
    st.markdown("### 📊 Statistics")
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        st.info("✅ Image loaded successfully!")
    else:
        st.warning("⏳ Please upload an image")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. **Upload** your image  
    2. **AI analyzes** color distribution  
    3. **Extract** dominant colors using K-means  
    4. **Download** your color palette  
    """)

# Main Content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        st.markdown(f"""
        <div class="metric-card">
            <strong>Image Info:</strong><br>
            📏 Size: {image.size[0]} x {image.size[1]} px<br>
            💾 File size: {len(uploaded_file.getvalue())/1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        st.markdown("### 🎨 Extracted Color Palette")
        with st.spinner("🔍 Analyzing colors..."):
            colors_with_percentages = extract_colors(image, n_colors)
            for i, (color, percentage) in enumerate(colors_with_percentages):
                hex_color = rgb_to_hex(color)
                text_color = get_text_color(color)
                st.markdown(f"""
                <div class="color-box" style="background-color: {hex_color}; color: {text_color};">
                    <strong>Color {i+1}</strong><br>
                    {hex_color}<br>
                    RGB({int(color[0])}, {int(color[1])}, {int(color[2])})<br>
                    {percentage:.1f}% of image
                </div>
                """, unsafe_allow_html=True)
            st.markdown("### 📥 Download Palette")
            palette_image = create_palette_image(colors_with_percentages)
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📊 Download Color Palette",
                    data=palette_image.getvalue(),
                    file_name=f"palette_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png"
                )
            with col_dl2:
                color_text = "Extracted Color Palette\n" + "="*25 + "\n"
                for i, (color, percentage) in enumerate(colors_with_percentages):
                    hex_color = rgb_to_hex(color)
                    color_text += f"Color {i+1}: {hex_color} | RGB({int(color[0])}, {int(color[1])}, {int(color[2])}) | {percentage:.1f}%\n"
                st.download_button(
                    label="📝 Download as Text",
                    data=color_text,
                    file_name=f"colors_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )

# Footer
st.markdown("""
<div class="footer-container">
    <h3>Made by Muhammad Zahran Muntazar - 140810230014</h3>
</div>
""", unsafe_allow_html=True)
