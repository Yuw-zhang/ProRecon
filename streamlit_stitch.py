
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import io
import json
import zipfile
 
def flip_matrix(w, h, flip_h, flip_v):
    """Same as flip_image(): PIL FLIP_LEFT_RIGHT first, then FLIP_TOP_BOTTOM."""
    F = np.eye(3)
    if flip_h:
        F = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], float) @ F
    if flip_v:
        F = np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], float) @ F
    return F
 
def load_features(npy_file, flip_h, flip_v):
    """Load an (H, W) or (H, W, C) feature matrix and flip it exactly like flip_image()."""
    if npy_file is None:
        return None
    feat = np.load(io.BytesIO(npy_file.getvalue())).astype(np.float32)
    if feat.ndim == 2:
        feat = feat[..., None]
    if flip_h:
        feat = feat[:, ::-1]
    if flip_v:
        feat = feat[::-1, :]
    return np.ascontiguousarray(feat)
 
def warp_features(feat, M, out_wh, categorical=False):
    """Warp every channel with the same 2x3 matrix used for the Gleason map."""
    flag = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    chans = [cv2.warpAffine(feat[..., c], M, out_wh, flags=flag, borderValue=np.nan)
             for c in range(feat.shape[2])]
    return np.stack(chans, axis=-1)
 
def flip_image(uploaded_file, flip_h, flip_v):
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file)
    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = np.array(img.convert("RGB"))
    return img
 
def get_majorTumor(msk):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(msk, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = np.zeros_like(msk)
    main_mask[labels == largest_label] = 255
 
    return main_mask
 
def get_mask(img):
    white = np.all(img == [255, 255, 255], axis=2)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[~white]=255
 
    return mask
 
def fit_circle_center(all_pts):
    if len(all_pts) < 3: return (0, 0)
    x = all_pts[:, 0]
    y = all_pts[:, 1]
 
    A_matrix = np.column_stack((x, y, np.ones(x.shape[0])))
    b_matrix = x**2 + y**2
 
    result, residues, rank, sing = np.linalg.lstsq(A_matrix, b_matrix, rcond=None)
    center_x = result[0] / 2
    center_y = result[1] / 2
 
    return (center_x, center_y)
 
def get_centroid(msk):
    moments = cv2.moments(msk)
    centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
 
    return centroid
 
def get_rectInfo(msk):
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box_width, box_height = rect[1]
    rect_angle = rect[2]
 
    if box_width >= box_height:
        length = box_width
        width = box_height
        raw_angle = rect_angle
    else:
        length = box_height
        width = box_width
        raw_angle = rect_angle + 90.0
 
    if raw_angle > 180: raw_angle -= 360
    elif raw_angle < -180: raw_angle += 360
 
    return raw_angle, length, width
 
def apply_tps_transform(image, src_points, dst_points):
    if len(src_points) < 4:
        return image
    
    src_pts = np.array([src_points], dtype=np.float32)
    dst_pts = np.array([dst_points], dtype=np.float32)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
 
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(dst_pts, src_pts, matches)
    return tps.warpImage(image)
 
def reset_stitch_state():
    st.session_state["has_stitched"] = False 
    
###############################LOAD TUMOR MAP###############################
st.title('PCa Reconstruction')
col1, col2 = st.columns(2)
with col1:
    # Panel for Img1: TopLeft
    img_tl = st.file_uploader("TopLeft", type=['png', 'jpg', 'jpeg'],on_change=reset_stitch_state)
    feat_tl = st.file_uploader("TopLeft features (.npy, optional)", type=['npy'], key="feat_tl", on_change=reset_stitch_state)
    flip_tl_h = st.checkbox("Horizontal Flip", key="tl_h")
    flip_tl_v = st.checkbox("Vertical Flip", key="tl_v")
    if img_tl is not None:
        preview_tl = flip_image(img_tl, flip_tl_h, flip_tl_v)
        st.image(preview_tl, caption='TopLeft view')
 
    # Panel for Img2: BottomLeft
    img_bl = st.file_uploader("BottomLeft", type=['png', 'jpg', 'jpeg'],on_change=reset_stitch_state)
    feat_bl = st.file_uploader("BottomLeft features (.npy, optional)", type=['npy'], key="feat_bl", on_change=reset_stitch_state)
    flip_bl_h = st.checkbox("Horizontal Flip", key="bl_h")
    flip_bl_v = st.checkbox("Vertical Flip", key="bl_v")
    if img_bl is not None:
        preview_bl = flip_image(img_bl, flip_bl_h, flip_bl_v)
        st.image(preview_bl, caption='BottomLeft view')
 
with col2:
    # Panel for Img3: TopRight
    img_tr = st.file_uploader("TopRight", type=['png', 'jpg', 'jpeg'],on_change=reset_stitch_state)
    feat_tr = st.file_uploader("TopRight features (.npy, optional)", type=['npy'], key="feat_tr", on_change=reset_stitch_state)
    flip_tr_h = st.checkbox("Horizontal Flip", key="tr_h")
    flip_tr_v = st.checkbox("Vertical Flip", key="tr_v")
    if img_tr is not None:
        preview_tr = flip_image(img_tr, flip_tr_h, flip_tr_v)
        st.image(preview_tr, caption='TopRight view')
 
    # Panel for Img4: BottomLeft
    img_br = st.file_uploader("BottomRight", type=['png', 'jpg', 'jpeg'],on_change=reset_stitch_state)
    feat_br = st.file_uploader("BottomRight features (.npy, optional)", type=['npy'], key="feat_br", on_change=reset_stitch_state)
    flip_br_h = st.checkbox("Horizontal Flip", key="br_h")
    flip_br_v = st.checkbox("Vertical Flip", key="br_v")
    if img_br is not None:
            preview_br = flip_image(img_br, flip_br_h, flip_br_v)
            st.image(preview_br, caption='BottomRight view')
###############################LOAD TUMOR MAP###############################
 
 
 
###############################START TO STITCH###############################
info_dict = {}
required_pos = [1, 2, 3, 4]
slide_canvas = None
if "has_stitched" not in st.session_state:
    st.session_state["has_stitched"] = False
 
if img_tl and img_bl and img_br and img_tr:
    col_img, col_ctrl = st.columns([2, 1])
 
    with col_ctrl:
        st.subheader("⚙️ Resction (Fine-tune Offsets)")  
        if st.button("Start", type="primary"):
            st.session_state["has_stitched"] = True
 
        if st.session_state["has_stitched"]:
            st.markdown("### 1. Coarse Rigid Fine-tune")
            gap_factor = st.slider("Gap Factor", min_value=0.500, max_value=1.500, value=0.950, step=0.001, key="gap_f")
            adj_col1, adj_col2 = st.columns(2)
            with adj_col1:
                st.markdown("**TopLeft**")
                dx1 = st.slider("Horizontal (X)", -100, 100, 0, key="dx1")
                dy1 = st.slider("Vertical (Y)", -100, 100, 0, key="dy1")
                da1 = st.slider("Angle (°)", -180, 180, 0, key="da1")
 
                st.markdown("**BottomLeft**")
                dx2 = st.slider("Horizontal (X)", -100, 100, 0, key="dx2")
                dy2 = st.slider("Vertical (Y)", -100, 100, 0, key="dy2")
                da2 = st.slider("Angle (°)", -180, 180, 0, key="da2")
            with adj_col2:
                st.markdown("**TopRight**")
                dx3 = st.slider("Horizontal (X)", -100, 100, 0, key="dx3")
                dy3 = st.slider("Vertical (Y)", -100, 100, 0, key="dy3")
                da3 = st.slider("Angle (°)", -180, 180, 0, key="da3")
 
                st.markdown("**BottomRight**")
                dx4 = st.slider("Horizontal (X)", -100, 100, 0, key="dx4")
                dy4 = st.slider("Vertical (Y)", -100, 100, 0, key="dy4")
                da4 = st.slider("Angle (°)", -180, 180, 0, key="da4")
 
            st.markdown("---")
            feat_categorical = st.checkbox("Feature channels are labels (use nearest-neighbor)", value=False)
        else:
            dx1 = dy1 = da1 = 0
            dx2 = dy2 = da2 = 0
            dx3 = dy3 = da3 = 0
            dx4 = dy4 = da4 = 0
            gap_factor = 0.95
            feat_categorical = False
 
    if st.session_state["has_stitched"]:
        # convert correctly flipped Tumor Map
        im1 = flip_image(img_tl, flip_tl_h, flip_tl_v)
        im2 = flip_image(img_bl, flip_bl_h, flip_bl_v)
        im3 = flip_image(img_tr, flip_tr_h, flip_tr_v)
        im4 = flip_image(img_br, flip_br_h, flip_br_v)
 
        # get the corresponding mask
        _, msk1 = cv2.threshold(get_majorTumor(get_mask(im1)), 1, 255, cv2.THRESH_BINARY)
        _, msk2 = cv2.threshold(get_majorTumor(get_mask(im2)), 1, 255, cv2.THRESH_BINARY)
        _, msk3 = cv2.threshold(get_majorTumor(get_mask(im3)), 1, 255, cv2.THRESH_BINARY)
        _, msk4 = cv2.threshold(get_majorTumor(get_mask(im4)), 1, 255, cv2.THRESH_BINARY)
 
        # get the centroid
        centroid1 = get_centroid(msk1)
        centroid2 = get_centroid(msk2)
        centroid3 = get_centroid(msk3)
        centroid4 = get_centroid(msk4)
 
        # get the bbox rect info
        angle1, h1, w1 = get_rectInfo(msk1)
        angle2, h2, w2 = get_rectInfo(msk2)
        angle3, h3, w3 = get_rectInfo(msk3)
        angle4, h4, w4 = get_rectInfo(msk4)
 
        info_dict[1] = {
            'mask': msk1,
            'map': im1,
            'centroid': centroid1,
            'raw_angle':angle1,
            'height':h1,
            'width': w1
        }
        info_dict[2] = {
            'mask': msk2,
            'map': im2,
            'centroid': centroid2,
            'raw_angle':angle2,
            'height':h2,
            'width': w2
        }
        info_dict[3] = {
            'mask': msk3,
            'map': im3,
            'centroid': centroid3,
            'raw_angle':angle3,
            'height':h3,
            'width': w3
        }
        info_dict[4] = {
            'mask': msk4,
            'map': im4,
            'centroid': centroid4,
            'raw_angle':angle4,
            'height':h4,
            'width': w4
        }
        if not all(pos in info_dict for pos in required_pos):
            print("Error: missing img info, stitching failed!")
 
        any_pos = required_pos[0]
        h, w = info_dict[any_pos]['mask'].shape[:2]
        slide_canvas = np.zeros((int(h*4), int(w*4), 3), dtype=np.uint8)
 
        canvas_center_x = slide_canvas.shape[1] // 2
        canvas_center_y = slide_canvas.shape[0] // 2
 
 
        left_height = (info_dict[1]['height']+ info_dict[1]['height'])/2
        right_height = (info_dict[2]['height']+ info_dict[2]['height'])/2
 
        top_width = (info_dict[3]['width']+ info_dict[3]['width'])/2
        bottom_width = (info_dict[4]['width']+ info_dict[4]['width'])/2
 
        offset_x_left = int((left_height / 2.0) * gap_factor)
        offset_x_right = int((right_height / 2.0) * gap_factor)
        offset_y_top = int((top_width / 2.0) * gap_factor)
        offset_y_bottom = int((bottom_width / 2.0) * gap_factor)
 
        quadrant_targets = {
            1: (canvas_center_x - offset_x_left + dx1, canvas_center_y - offset_y_top + dy1),
            2: (canvas_center_x - offset_x_left + dx2, canvas_center_y + offset_y_bottom + dy2),
            3: (canvas_center_x + offset_x_right + dx3, canvas_center_y - offset_y_top + dy3),
            4: (canvas_center_x + offset_x_right + dx4, canvas_center_y + offset_y_bottom + dy4)
        }
 
        delta_angles = {1:da1, 2:da2, 3:da3, 4:da4}
 
        pos_names = {1: "TopLeft", 2: "BottomLeft", 3: "TopRight", 4: "BottomRight"}
        uploads = {1: img_tl, 2: img_bl, 3: img_tr, 4: img_br}
        feat_uploads = {1: feat_tl, 2: feat_bl, 3: feat_tr, 4: feat_br}
        flips = {1: (flip_tl_h, flip_tl_v), 2: (flip_bl_h, flip_bl_v),
                 3: (flip_tr_h, flip_tr_v), 4: (flip_br_h, flip_br_v)}
        offsets = {1: (dx1, dy1), 2: (dx2, dy2), 3: (dx3, dy3), 4: (dx4, dy4)}
        transform_log = {
            "canvas_size_hw": [int(slide_canvas.shape[0]), int(slide_canvas.shape[1])],
            "gap_factor": float(gap_factor),
            "coordinate_convention": "pixel (x, y, 1); canvas_xy = M @ original_xy",
            "quadrants": {},
        }
 
        # load features (flipped like the maps) and check they match the Gleason map grid
        feats = {}
        for pos in required_pos:
            f = load_features(feat_uploads[pos], *flips[pos])
            if f is not None and f.shape[:2] != info_dict[pos]['map'].shape[:2]:
                st.error(f"{pos_names[pos]}: feature matrix {f.shape[:2]} does not match Gleason map "
                         f"{info_dict[pos]['map'].shape[:2]}; it will be skipped.")
                f = None
            feats[pos] = f
        n_ch = {f.shape[2] for f in feats.values() if f is not None}
        if len(n_ch) > 1:
            st.error(f"Feature matrices have different numbers of channels: {sorted(n_ch)}")
        feature_canvas = None
        if len(n_ch) == 1:
            feature_canvas = np.full((slide_canvas.shape[0], slide_canvas.shape[1], n_ch.pop()),
                                     np.nan, np.float32)
 
        for pos in required_pos:
            data = info_dict[pos]
            msk = data['mask']
            map = data['map']
            centroid = data['centroid']
            raw_angle = data['raw_angle']
 
            target_angle = 90
            rotation_needed = raw_angle + delta_angles[pos]
 
            M_rot = cv2.getRotationMatrix2D(centroid, rotation_needed, 1.0)
 
            target_x, target_y = quadrant_targets[pos]
            M_rot[0, 2] += (target_x - centroid[0])
            M_rot[1, 2] += (target_y - centroid[1])
 
            transformed_map = cv2.warpAffine(map, M_rot, (slide_canvas.shape[1], slide_canvas.shape[0]))
            transformed_mask = cv2.warpAffine(msk, M_rot, (slide_canvas.shape[1], slide_canvas.shape[0]))
 
            slide_canvas[transformed_mask > 0] = transformed_map[transformed_mask > 0]
 
            # same transform for the feature matrix
            if feature_canvas is not None and feats[pos] is not None:
                wf = warp_features(feats[pos], M_rot, (slide_canvas.shape[1], slide_canvas.shape[0]),
                                   categorical=feat_categorical)
                feature_canvas[transformed_mask > 0] = wf[transformed_mask > 0]
 
            # full transform from the ORIGINAL (unflipped) uploaded image to the canvas
            h0, w0 = map.shape[:2]
            F = flip_matrix(w0, h0, *flips[pos])
            M_total = np.vstack([M_rot, [0, 0, 1]]) @ F
            transform_log["quadrants"][pos_names[pos]] = {
                "source_file": uploads[pos].name,
                "feature_file": feat_uploads[pos].name if feat_uploads[pos] else None,
                "original_size_hw": [int(h0), int(w0)],
                "flip_h": bool(flips[pos][0]),
                "flip_v": bool(flips[pos][1]),
                "auto_angle_deg": float(raw_angle),
                "manual_angle_deg": float(delta_angles[pos]),
                "rotation_deg": float(rotation_needed),
                "rotation_center_xy": [float(centroid[0]), float(centroid[1])],
                "manual_offset_xy": [int(offsets[pos][0]), int(offsets[pos][1])],
                "target_xy": [float(target_x), float(target_y)],
                "M": M_total.tolist(),
            }
 
        result_img = Image.fromarray(slide_canvas)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
 
        with col_img:
            st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
            st.subheader("Reconstructed specimen")
            st.image(slide_canvas, caption="", use_container_width=True)
            
            custom_filename = st.text_input("Saved Filename", value="reconstructed")
            if not custom_filename.endswith(".png"):
                custom_filename += ".png"
                
            # Download PNG Button
            st.download_button(
                label='Download PNG',
                data=byte_im,
                file_name=custom_filename,
                mime='image/png',
                type='primary'
            )
 
            # Download PNG + transforms (+ stitched features) as one ZIP
            base = custom_filename[:-4]
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(custom_filename, byte_im)
                z.writestr(f"{base}_transforms.json", json.dumps(transform_log, indent=2))
                if feature_canvas is not None:
                    nbuf = io.BytesIO()
                    np.save(nbuf, feature_canvas)
                    z.writestr(f"{base}_features.npy", nbuf.getvalue())
            st.download_button(
                label='Download PNG + transforms + features (ZIP)',
                data=zbuf.getvalue(),
                file_name=f"{base}.zip",
                mime='application/zip'
            )
    else:
        with col_img:
            st.info("Please upload maps and click \'Start\'")
