import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import io
from collections import Counter

# Konfigurasi awal halaman
st.set_page_config(
    page_title="🎨 Palette Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling tampilan
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23, #1a1a2e, #16213e);
        color: #fff;
    }
    .main .block-container {
        background: transparent;
    }
    .header {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
    }
    .header h1 {
        font-size: 2.8rem;
        margin: 0;
    }
    .color-box {
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        text-align: center;
        font-weight: bold;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        padding: 1.5rem;
        background: #1e293b;
        border-radius: 15px;
        color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi bantu
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(map(int, rgb))

def get_text_color(rgb):
    brightness = (rgb[0]*299 + rgb[1]*587 + rgb[2]*114) / 1000
    return 'black' if brightness > 128 else 'white'

def extract_colors(img, k=5):
    arr = np.array(img).reshape(-1, 3)
    arr = arr[~np.all(arr < 30, axis=1)]
    arr = arr[~np.all(arr > 225, axis=1)]
    if len(arr) == 0:
        arr = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(arr)
    colors = kmeans.cluster_centers_
    counts = Counter(kmeans.labels_)
    total = len(kmeans.labels_)
    palette = [(colors[i], (counts[i]/total)*100) for i in range(k)]
    return sorted(palette, key=lambda x: x[1], reverse=True)

def generate_palette_img(palette):
    fig, (bar_ax, pie_ax) = plt.subplots(1, 2, figsize=(14, 6))
    colors = [c for c, _ in palette]
    percentages = [p for _, p in palette]
    y = range(len(colors))
    bars = bar_ax.barh(y, [1]*len(colors), color=[np.array(c)/255 for c in colors])
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels([f'Color {i+1}' for i in y])
    bar_ax.set_xlim(0, 1)
    bar_ax.set_title("Dominant Colors", fontsize=15)
    for i, bar in enumerate(bars):
        hex_ = rgb_to_hex(colors[i])
        color = 'white' if sum(colors[i]) < 400 else 'black'
        bar_ax.text(0.5, i, f'{hex_}\n{percentages[i]:.1f}%', ha='center', va='center', color=color)
    pie_ax.pie(percentages, colors=[np.array(c)/255 for c in colors], labels=[rgb_to_hex(c) for c in colors], autopct='%1.1f%%')
    pie_ax.set_title("Color Share", fontsize=15)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    return buf

# Header
st.markdown("""
<div class="header">
    <h1>🎨 Color Palette Extractor</h1>
    <p>Upload gambar dan temukan warna dominan dalam hitungan detik</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    n_colors = st.slider("Jumlah warna", 3, 8, 5)
    st.markdown("#### Cara Kerja")
    st.markdown("""
    1. Upload gambar  
    2. Sistem menganalisis warna  
    3. Warna dominan diekstrak  
    4. Unduh palet warna
    """)

# Main Area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Gambar")
    file = st.file_uploader("Pilih file gambar", type=['jpg', 'jpeg', 'png', 'bmp'])
    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption=file.name, use_container_width=True)
        st.info(f"Ukuran gambar: {image.size[0]}x{image.size[1]} px | {len(file.getvalue())/1024:.1f} KB")

with col2:
    if file:
        st.subheader("🎨 Warna Dominan")
        with st.spinner("Menganalisis warna..."):
            result = extract_colors(image, n_colors)
            for idx, (color, percent) in enumerate(result):
                hex_code = rgb_to_hex(color)
                text_col = get_text_color(color)
                st.markdown(f"""
                <div class="color-box" style="background-color:{hex_code};color:{text_col}">
                    <strong>Color {idx+1}</strong><br>
                    {hex_code}<br>
                    RGB({int(color[0])}, {int(color[1])}, {int(color[2])})<br>
                    {percent:.1f}%
                </div>
                """, unsafe_allow_html=True)
            img_palette = generate_palette_img(result)
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button("📊 Unduh Palet", img_palette.getvalue(), file_name=f"palette_{file.name}.png", mime="image/png")
            with col_b:
                text = "\n".join([f"Color {i+1}: {rgb_to_hex(c)} | RGB({int(c[0])}, {int(c[1])}, {int(c[2])}) | {p:.1f}%" for i, (c, p) in enumerate(result)])
                st.download_button("📝 Simpan ke TXT", text, file_name=f"colors_{file.name}.txt")

# Footer
st.markdown("""
<div class="footer">
    <h4>Dibuat oleh Daniel Bintang W. Sitorus - 140810230048</h4>
</div>
""", unsafe_allow_html=True)
