import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import tempfile
import cairosvg
from pdf2image import convert_from_path
import os

st.set_page_config(page_title="SVG Layer Editor", layout="wide")

# --- Utils ---
@st.cache_data
def load_input_image(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == '.pdf':
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(uploaded_file.read())
            tf.flush()
            pages = convert_from_path(tf.name, dpi=300)
            pil = pages[0].convert('RGBA')
    else:
        pil = Image.open(uploaded_file).convert('RGBA')
    return pil

def segment_image(pil_image, min_area=300):
    arr = np.array(pil_image.convert('L'))
    h, w = arr.shape
    blur = cv2.GaussianBlur(arr, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    layers = []
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, wc, hc = cv2.boundingRect(c)
        aspect = wc / float(hc + 1)
        ltype = 'text' if (area < 5000 and aspect > 2 or aspect < 0.5) else 'image'
        layers.append({
            'id': f'layer_{idx}', 'name': f'{ltype}_{idx}', 'type': ltype,
            'bbox': (x, y, wc, hc), 'contour': c.tolist(), 'area': int(area),
            'visible': True, 'color': '#000000'
        })
    if not any(l['name'] == 'background' for l in layers):
        layers.insert(0, {'id': 'layer_bg', 'name': 'background', 'type': 'bg', 'bbox': (0, 0, w, h),
                          'contour': np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]).tolist(),
                          'area': w * h, 'visible': True, 'color': '#ffffff'})
    return layers

def contour_to_path(contour):
    pts = np.array(contour).reshape(-1,2)
    d = f"M {pts[0][0]} {pts[0][1]} " + " ".join([f"L {x} {y}" for x, y in pts[1:]]) + " Z"
    return d

def build_svg(pil_image, layers, image_path):
    w, h = pil_image.size
    rel_image = Path(image_path).name
    svg = [f'<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append('<defs>')
    for l in layers:
        if not l['visible']: continue
        lid = l['id']
        pathd = contour_to_path(l['contour'])
        svg.append(f'<clipPath id="cp_{lid}"><path d="{pathd}"/></clipPath>')
    svg.append('</defs>')
    for l in layers:
        if not l['visible']: continue
        lid, name = l['id'], l['name']
        color = l.get('color', '#000')
        pathd = contour_to_path(l['contour'])
        svg.append(f'<g id="{lid}" data-name="{name}" data-type="{l["type"]}" data-visible="{str(l["visible"]).lower()}">')
        svg.append(f'  <g clip-path="url(#cp_{lid})">')
        svg.append(f'    <image href="{rel_image}" x="0" y="0" width="{w}" height="{h}"/>')
        svg.append('  </g>')
        svg.append(f'  <path d="{pathd}" fill="{color}" fill-opacity="0.0" stroke="none"/>')
        svg.append('</g>')
    svg.append('</svg>')
    return "\n".join(svg)

def render_svg_to_png(svg_content, width, height):
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tf_svg:
        tf_svg.write(svg_content.encode('utf-8'))
        tf_svg.flush()
        png_path = tf_svg.name.replace('.svg', '.png')
        cairosvg.svg2png(url=tf_svg.name, write_to=png_path, output_width=width, output_height=height)
        img = Image.open(png_path)
    return img

# --- App ---
st.title("SVG Layer Segmentation & Editor")

uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG, or PDF)", type=["png", "jpg", "jpeg", "pdf"])
if uploaded_file:
    pil_img = load_input_image(uploaded_file)
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, "source.png")
    pil_img.save(image_path, "PNG")
    layers = segment_image(pil_img)
    if "layers_state" not in st.session_state:
        st.session_state["layers_state"] = layers
    else:
        layers = st.session_state["layers_state"]

    cols = st.columns([1, 2])
    with cols[0]:
        st.subheader("Layers")
        for idx, l in enumerate(layers):
            col1, col2, col3 = st.columns([2,1,1])
            with col1:
                st.checkbox(f"{l['name']} ({l['type']})", value=l["visible"], key=f"layer_{idx}", on_change=None, disabled=False)
            with col2:
                if st.button("Hide" if l["visible"] else "Show", key=f"toggle_{idx}"):
                    l["visible"] = not l["visible"]
            with col3:
                if st.button("Remove", key=f"remove_{idx}"):
                    l["visible"] = False
        st.session_state["layers_state"] = layers
        st.markdown("---")
        if st.button("Reset layers"):
            st.session_state["layers_state"] = segment_image(pil_img)
            st.experimental_rerun()
    with cols[1]:
        st.subheader("Preview")
        svg_content = build_svg(pil_img, layers, image_path)
        preview_img = render_svg_to_png(svg_content, *pil_img.size)
        st.image(preview_img, caption="SVG Preview", use_column_width=True)
        st.download_button("Download SVG", svg_content, file_name="output.svg", mime="image/svg+xml")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            preview_img.save(tf.name, "PNG")
            tf.flush()
            st.download_button("Download PNG", tf.read(), file_name="output.png", mime="image/png")