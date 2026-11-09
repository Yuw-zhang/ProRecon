
"""Section picker: choose one tissue section from a whole-slide Gleason map and crop
the Gleason map and its feature matrix with the same box.
 
Run:  streamlit run section_picker.py
 
Inputs (same grid, i.e., same H x W):
  - Gleason map PNG of the whole slide (background = white)
  - Feature matrix .npy, shape (H, W) or (H, W, C)  [optional]
 
Outputs (download as one ZIP):
  - <name>_sectionN.png        cropped Gleason map (upload this to the stitching app)
  - <name>_sectionN.npy        cropped feature matrix (pixels outside the section = NaN)
  - <name>_sectionN_crop.json  crop box + 3x3 matrix C mapping whole-slide grid -> cropped grid
"""
import io
import json
import zipfile
 
import cv2
import numpy as np
import streamlit as st
from PIL import Image
 
 
# ---------------- core functions (no Streamlit) ----------------
def tissue_mask(rgb):
    """Same rule as the stitching app: anything that is not pure white is tissue."""
    return ~np.all(rgb == [255, 255, 255], axis=2)
 
 
def find_sections(rgb, min_area=200):
    """Connected tissue components, sorted left to right. Returns label image and a list of dicts."""
    m = tissue_mask(rgb).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = stats[k, :4]
        comps.append({"label": int(k), "area": int(area), "bbox": [int(x), int(y), int(w), int(h)],
                      "centroid": [float(cent[k, 0]), float(cent[k, 1])]})
    comps.sort(key=lambda c: c["centroid"][0])          # left to right
    for i, c in enumerate(comps, 1):
        c["section"] = i
    return lab, comps
 
 
def crop_section(rgb, feat, lab, chosen_labels, pad=5, keep_all_in_box=False):
    """Crop Gleason map and features with the union bbox of the chosen components."""
    sel = np.isin(lab, chosen_labels)
    ys, xs = np.where(sel)
    H, W = sel.shape
    x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
    x1, y1 = min(int(xs.max()) + pad + 1, W), min(int(ys.max()) + pad + 1, H)
 
    keep = np.ones_like(sel) if keep_all_in_box else sel
    keep = keep[y0:y1, x0:x1]
 
    rgb_c = rgb[y0:y1, x0:x1].copy()
    rgb_c[~keep] = 255                                    # other tissue -> white background
 
    feat_c = None
    if feat is not None:
        feat_c = feat[y0:y1, x0:x1].astype(np.float32).copy()
        feat_c[~keep] = np.nan                            # other tissue -> NaN
 
    C = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)   # whole-slide -> cropped coords
    box = {"x0": x0, "y0": y0, "width": x1 - x0, "height": y1 - y0}
    return rgb_c, feat_c, box, C
 
 
def label_overlay(rgb, comps):
    """Draw section numbers and boxes for display."""
    vis = rgb.copy()
    for c in comps:
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cx, cy = map(int, c["centroid"])
        cv2.putText(vis, str(c["section"]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    max(1.0, w / 200), (255, 0, 0), 3, cv2.LINE_AA)
    return vis
 
 
# ---------------- Streamlit UI ----------------
def main():
    st.title("Section picker")
    png_file = st.file_uploader("Whole-slide Gleason map (PNG)", type=["png"])
    npy_file = st.file_uploader("Feature matrix (.npy, same grid; optional)", type=["npy"])
    if png_file is None:
        st.info("Upload the whole-slide Gleason map to start.")
        return
 
    name = png_file.name.rsplit(".", 1)[0]
    rgb = np.array(Image.open(png_file).convert("RGB"))
    feat = None
    if npy_file is not None:
        feat = np.load(io.BytesIO(npy_file.getvalue()))
        if feat.shape[:2] != rgb.shape[:2]:
            st.error(f"Feature matrix {feat.shape[:2]} and Gleason map {rgb.shape[:2]} are not on the same grid.")
            return
 
    min_area = st.number_input("Ignore fragments smaller than (pixels)", 0, 100000, 200, 50)
    lab, comps = find_sections(rgb, min_area)
    if not comps:
        st.error("No tissue found.")
        return
 
    st.image(label_overlay(rgb, comps), caption="Sections numbered left to right",
             use_container_width=True)
    st.write(f"{len(comps)} section(s) found:",
             {c["section"]: f'area {c["area"]} px' for c in comps})
 
    chosen = st.multiselect("Section(s) to keep", [c["section"] for c in comps],
                            default=[max(comps, key=lambda c: c["area"])["section"]])
    keep_all = st.checkbox("Keep everything inside the box (including small fragments of other labels)",
                           value=False)
    pad = st.slider("Padding (pixels)", 0, 50, 5)
    if not chosen:
        st.warning("Choose at least one section.")
        return
 
    chosen_labels = [c["label"] for c in comps if c["section"] in chosen]
    rgb_c, feat_c, box, C = crop_section(rgb, feat, lab, chosen_labels, pad, keep_all)
    st.image(rgb_c, caption=f"Cropped section(s) {chosen}  box={box}")
 
    tag = f"{name}_section{'-'.join(map(str, chosen))}"
    info = {"source_png": png_file.name,
            "source_features": npy_file.name if npy_file else None,
            "source_size_hw": list(rgb.shape[:2]),
            "sections_kept": chosen,
            "keep_all_in_box": keep_all,
            "crop_box": box,
            "C_wholeslide_to_crop": C.tolist(),
            "note": "canvas_xy = M_stitch @ C @ wholeslide_xy"}
 
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        pbuf = io.BytesIO(); Image.fromarray(rgb_c).save(pbuf, format="PNG")
        z.writestr(f"{tag}.png", pbuf.getvalue())
        if feat_c is not None:
            nbuf = io.BytesIO(); np.save(nbuf, feat_c)
            z.writestr(f"{tag}.npy", nbuf.getvalue())
        z.writestr(f"{tag}_crop.json", json.dumps(info, indent=2))
 
    st.download_button("Download cropped PNG + features + crop info (ZIP)", zbuf.getvalue(),
                       file_name=f"{tag}.zip", mime="application/zip", type="primary")
 
 
if __name__ == "__main__":
    main()
