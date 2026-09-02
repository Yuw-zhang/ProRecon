import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

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
    if num_labels <= 1:
        return msk
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = np.zeros_like(msk)
    main_mask[labels == largest_label] = 255
    return main_mask

def get_mask(img):
    white = np.all(img == [255, 255, 255], axis=2)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[~white] = 255
    return mask

def get_auto_flatten_angle(mask, pos):
    """
    Find the inner straight cut edges and calculate rotation needed to make them flat (0° / 90°).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (cx, cy), (width, height), angle = rect

    # Normalize minimum area rectangle angle
    if width < height:
        angle = angle - 90.0
    
    # Keep angle within [-45, 45] range to avoid unexpected upside-down flips
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0

    return -angle

def get_inner_corner_offset(mask, pos):
    """ Locate the inner corner point of the tissue slice facing towards the canvas center. """
    pts = cv2.findNonZero(mask)
    if pts is None:
        return mask.shape[1] // 2, mask.shape[0] // 2
    
    pts = pts.reshape(-1, 2)
    
    if pos == 1:   # TopLeft -> Corner facing Bottom-Right
        idx = np.argmax(pts[:, 0] + pts[:, 1])
    elif pos == 2: # BottomLeft -> Corner facing Top-Right
        idx = np.argmax(pts[:, 0] - pts[:, 1])
    elif pos == 3: # TopRight -> Corner facing Bottom-Left
        idx = np.argmax(-pts[:, 0] + pts[:, 1])
    elif pos == 4: # BottomRight -> Corner facing Top-Left
        idx = np.argmin(pts[:, 0] + pts[:, 1])
    
    return pts[idx][0], pts[idx][1]

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

############################### LOAD TUMOR MAP ###############################
st.title('PCa Reconstruction')
col1, col2 = st.columns(2)
with col1:
    img_tl = st.file_uploader("TopLeft", type=['png', 'jpg', 'jpeg'], on_change=reset_stitch_state)
    flip_tl_h = st.checkbox("Horizontal Flip", key="tl_h")
    flip_tl_v = st.checkbox("Vertical Flip", key="tl_v")
    if img_tl is not None:
        preview_tl = flip_image(img_tl, flip_tl_h, flip_tl_v)
        st.image(preview_tl, caption='TopLeft view')

    img_bl = st.file_uploader("BottomLeft", type=['png', 'jpg', 'jpeg'], on_change=reset_stitch_state)
    flip_bl_h = st.checkbox("Horizontal Flip", key="bl_h")
    flip_bl_v = st.checkbox("Vertical Flip", key="bl_v")
    if img_bl is not None:
        preview_bl = flip_image(img_bl, flip_bl_h, flip_bl_v)
        st.image(preview_bl, caption='BottomLeft view')

with col2:
    img_tr = st.file_uploader("TopRight", type=['png', 'jpg', 'jpeg'], on_change=reset_stitch_state)
    flip_tr_h = st.checkbox("Horizontal Flip", key="tr_h")
    flip_tr_v = st.checkbox("Vertical Flip", key="tr_v")
    if img_tr is not None:
        preview_tr = flip_image(img_tr, flip_tr_h, flip_tr_v)
        st.image(preview_tr, caption='TopRight view')

    img_br = st.file_uploader("BottomRight", type=['png', 'jpg', 'jpeg'], on_change=reset_stitch_state)
    flip_br_h = st.checkbox("Horizontal Flip", key="br_h")
    flip_br_v = st.checkbox("Vertical Flip", key="br_v")
    if img_br is not None:
        preview_br = flip_image(img_br, flip_br_h, flip_br_v)
        st.image(preview_br, caption='BottomRight view')
############################### LOAD TUMOR MAP ###############################


############################### START TO STITCH ###############################
required_pos = [1, 2, 3, 4]
slide_canvas = None

if "has_stitched" not in st.session_state:
    st.session_state["has_stitched"] = False

if img_tl and img_bl and img_br and img_tr:
    col_img, col_ctrl = st.columns([2, 1])

    with col_ctrl:
        st.subheader("⚙️ Resection (Fine-tune Offsets)")  
        auto_flatten = st.checkbox("Enable Auto-Flattening Cut Edges", value=True)
        if st.button("Start", type="primary"):
            st.session_state["has_stitched"] = True

        if st.session_state["has_stitched"]:
            st.markdown("### 1. Coarse Rigid Fine-tune")
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
            st.markdown("### 2. TPS Non-Rigid Warping")
            enable_tps = st.checkbox("Enable TPS Boundary Stitching", value=False)
            tps_strength = st.slider("Stitching Tension Strength", -50, 50, 0, key="tps_s")
        else:
            dx1 = dy1 = da1 = 0
            dx2 = dy2 = da2 = 0
            dx3 = dy3 = da3 = 0
            dx4 = dy4 = da4 = 0
            enable_tps = False
            tps_strength = 0

    if st.session_state["has_stitched"]:
        # 1. Preprocess images
        im1 = flip_image(img_tl, flip_tl_h, flip_tl_v)
        im2 = flip_image(img_bl, flip_bl_h, flip_bl_v)
        im3 = flip_image(img_tr, flip_tr_h, flip_tr_v)
        im4 = flip_image(img_br, flip_br_h, flip_br_v)

        msk1 = get_mask(im1)
        msk2 = get_mask(im2)
        msk3 = get_mask(im3)
        msk4 = get_mask(im4)

        info_dict = {
            1: {'map': im1, 'mask': msk1, 'dx': dx1, 'dy': dy1, 'da': da1},
            2: {'map': im2, 'mask': msk2, 'dx': dx2, 'dy': dy2, 'da': da2},
            3: {'map': im3, 'mask': msk3, 'dx': dx3, 'dy': dy3, 'da': da3},
            4: {'map': im4, 'mask': msk4, 'dx': dx4, 'dy': dy4, 'da': da4},
        }

        # 2. Setup canvas
        max_h = max(im.shape[0] for im in [im1, im2, im3, im4])
        max_w = max(im.shape[1] for im in [im1, im2, im3, im4])
        slide_canvas = np.zeros((max_h * 3, max_w * 3, 3), dtype=np.uint8)

        canvas_cx = slide_canvas.shape[1] // 2
        canvas_cy = slide_canvas.shape[0] // 2

        # 3. Process each slice
        for pos in required_pos:
            item = info_dict[pos]
            map_img = item['map']
            msk = item['mask']
            
            img_h, img_w = map_img.shape[:2]
            
            # Pad canvas to prevent cropping during rotation
            pad = max(img_h, img_w)
            temp_h, temp_w = img_h + 2 * pad, img_w + 2 * pad
            
            temp_map = np.zeros((temp_h, temp_w, 3), dtype=np.uint8)
            temp_mask = np.zeros((temp_h, temp_w), dtype=np.uint8)
            
            temp_map[pad:pad+img_h, pad:pad+img_w] = map_img
            temp_mask[pad:pad+img_h, pad:pad+img_w] = msk
            
            temp_cx, temp_cy = pad + img_w / 2.0, pad + img_h / 2.0

            # Calculate rotation: Auto-flatten angle + manual slider angle (da)
            auto_angle = get_auto_flatten_angle(msk, pos) if auto_flatten else 0.0
            total_angle = auto_angle + item['da']
            
            # Rotate slice around center
            M_rot = cv2.getRotationMatrix2D((temp_cx, temp_cy), total_angle, 1.0)
            rotated_map = cv2.warpAffine(temp_map, M_rot, (temp_w, temp_h))
            rotated_mask = cv2.warpAffine(temp_mask, M_rot, (temp_w, temp_h))

            # Locate the inner corner on rotated mask
            corner_x, corner_y = get_inner_corner_offset(rotated_mask, pos)

            # Align inner corner to canvas center with manual dx/dy offset
            M_trans = np.float32([
                [1, 0, (canvas_cx - corner_x) + item['dx']],
                [0, 1, (canvas_cy - corner_y) + item['dy']]
            ])

            transformed_map = cv2.warpAffine(rotated_map, M_trans, (slide_canvas.shape[1], slide_canvas.shape[0]))
            transformed_mask = cv2.warpAffine(rotated_mask, M_trans, (slide_canvas.shape[1], slide_canvas.shape[0]))

            # Paste into canvas
            slide_canvas[transformed_mask > 0] = transformed_map[transformed_mask > 0]

        # 4. Apply TPS transform if enabled
        if enable_tps:
            ch, cw = slide_canvas.shape[:2]
            cx, cy = cw // 2, ch // 2

            src_pts = [
                [0, 0], [cw, 0], [0, ch], [cw, ch],             
                [cx, cy - 100], [cx, cy + 100], [cx - 100, cy], [cx + 100, cy] 
            ]
            
            dst_pts = [
                [0, 0], [cw, 0], [0, ch], [cw, ch],
                [cx, cy - 100 + tps_strength], 
                [cx, cy + 100 - tps_strength], 
                [cx - 100 + tps_strength, cy], 
                [cx + 100 - tps_strength, cy]
            ]
            
            slide_canvas = apply_tps_transform(slide_canvas, src_pts, dst_pts)

        # 5. Export PNG download
        result_img = Image.fromarray(slide_canvas)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        with col_img:
            st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
            st.subheader("Reconstructed specimen")
            st.image(slide_canvas, caption="", use_container_width=True)
            
            st.download_button(
                label='Download PNG',
                data=byte_im,
                file_name='reconstructed.png',
                mime='image/png',
                type='primary'
            )
    else:
        with col_img:
            st.info("Please upload maps and click 'Start'")
