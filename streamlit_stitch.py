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

def get_mask(img):
    """ Extract tissue mask based on non-white pixels """
    white = np.all(img == [255, 255, 255], axis=2)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[~white] = 255
    return mask

def get_centroid(msk):
    """ Calculate centroid of tissue mask """
    moments = cv2.moments(msk)
    if moments["m00"] == 0:
        return (msk.shape[1] / 2, msk.shape[0] / 2)
    centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    return centroid

def get_auto_rectify_angle(msk, pos):
    """
    Calculate the rotation angle required to rectify (straighten) the slice 
    based on the minimum area bounding box of the tissue mask.
    """
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box_width, box_height = rect[1]
    rect_angle = rect[2]

    # Normalize angle based on aspect ratio
    if box_width < box_height:
        raw_angle = rect_angle - 90.0
    else:
        raw_angle = rect_angle

    if raw_angle > 180: raw_angle -= 360
    elif raw_angle < -180: raw_angle += 360

    return raw_angle

def get_inner_corner_offset(mask, pos):
    """
    Find the inner corner point of the tissue slice facing the canvas center.
    Position 1 (TopLeft):      Finds bottom-rightmost point (Max X, Max Y)
    Position 2 (BottomLeft):   Finds top-rightmost point (Max X, Min Y)
    Position 3 (TopRight):     Finds bottom-leftmost point (Min X, Max Y)
    Position 4 (BottomRight):  Finds top-leftmost point (Min X, Min Y)
    """
    pts = cv2.findNonZero(mask)
    if pts is None:
        return mask.shape[1] // 2, mask.shape[0] // 2
    
    pts = pts.reshape(-1, 2)
    
    if pos == 1:   # TopLeft -> Corner at bottom-right
        idx = np.argmax(pts[:, 0] + pts[:, 1])
    elif pos == 2: # BottomLeft -> Corner at top-right
        idx = np.argmax(pts[:, 0] - pts[:, 1])
    elif pos == 3: # TopRight -> Corner at bottom-left
        idx = np.argmax(-pts[:, 0] + pts[:, 1])
    elif pos == 4: # BottomRight -> Corner at top-left
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
    # Panel for Img1: TopLeft
    img_tl = st.file_uploader("TopLeft", type=['png', 'jpg', 'jpeg'], on_change=reset_stitch_state)
    flip_tl_h = st.checkbox("Horizontal Flip", key="tl_h")
    flip_tl_v = st.checkbox("Vertical Flip", key="tl_v")
    if img_tl is not None:
        preview_tl = flip_image(img_tl, flip_tl_h, flip_tl_v)
        st.image(preview_tl, caption='TopLeft view')

    # Panel for Img2: BottomLeft
    img_bl = st.file_uploader("BottomLeft", type=['png', 'jpg', 'jpeg'], on_change=reset_stitch_state)
    flip_bl_h = st.checkbox("Horizontal Flip", key="bl_h")
    flip_bl_v = st.checkbox("Vertical Flip", key="bl_v")
    if img_bl is not None:
        preview_bl = flip_image(img_bl, flip_bl_h, flip_bl_v)
        st.image(preview_bl, caption='BottomLeft view')

with col2:
    # Panel for Img3: TopRight
    img_tr = st.file_uploader("TopRight", type=['png', 'jpg', 'jpeg'], on_change=reset_stitch_state)
    flip_tr_h = st.checkbox("Horizontal Flip", key="tr_h")
    flip_tr_v = st.checkbox("Vertical Flip", key="tr_v")
    if img_tr is not None:
        preview_tr = flip_image(img_tr, flip_tr_h, flip_tr_v)
        st.image(preview_tr, caption='TopRight view')

    # Panel for Img4: BottomRight
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
        # 1. Load and apply flip transformations
        im1 = flip_image(img_tl, flip_tl_h, flip_tl_v)
        im2 = flip_image(img_bl, flip_bl_h, flip_bl_v)
        im3 = flip_image(img_tr, flip_tr_h, flip_tr_v)
        im4 = flip_image(img_br, flip_br_h, flip_br_v)

        # 2. Extract gray tissue mask
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

        # 3. Create canvas based on maximum dimensions
        max_h = max(im.shape[0] for im in [im1, im2, im3, im4])
        max_w = max(im.shape[1] for im in [im1, im2, im3, im4])
        slide_canvas = np.zeros((max_h * 3, max_w * 3, 3), dtype=np.uint8)

        canvas_cx = slide_canvas.shape[1] // 2
        canvas_cy = slide_canvas.shape[0] // 2

        # 4. Two-Step Alignment: (Step 1: Rectify/Straighten -> Step 2: Inner-Corner Snap)
        for pos in required_pos:
            item = info_dict[pos]
            map_img = item['map']
            msk = item['mask']
            
            # --- STEP 1: Calculate Centroid / Rectify Angle to Straighten the Slice ---
            centroid_x, centroid_y = get_centroid(msk)
            auto_angle = get_auto_rectify_angle(msk, pos)
            total_angle = auto_angle + item['da']

            # Rotate slice around its centroid to rectify orientation
            M_rectify = cv2.getRotationMatrix2D((centroid_x, centroid_y), total_angle, 1.0)
            rectified_map = cv2.warpAffine(map_img, M_rectify, (map_img.shape[1], map_img.shape[0]))
            rectified_mask = cv2.warpAffine(msk, M_rectify, (msk.shape[1], msk.shape[0]))

            # --- STEP 2: Find Inner Cutting Corner AFTER Rectification & Snap to Canvas Center ---
            corner_x, corner_y = get_inner_corner_offset(rectified_mask, pos)

            # Translation matrix: Align rectified inner corner directly to canvas center
            M_translate = np.float32([
                [1, 0, (canvas_cx - corner_x) + item['dx']],
                [0, 1, (canvas_cy - corner_y) + item['dy']]
            ])

            # Apply final translation to canvas
            transformed_map = cv2.warpAffine(rectified_map, M_translate, (slide_canvas.shape[1], slide_canvas.shape[0]))
            transformed_mask = cv2.warpAffine(rectified_mask, M_translate, (slide_canvas.shape[1], slide_canvas.shape[0]))

            # Blend onto canvas
            slide_canvas[transformed_mask > 0] = transformed_map[transformed_mask > 0]

        # 5. TPS non-rigid warping (optional edge deformation)
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

        # 6. Convert slide_canvas to BytesIO for download button
        result_img = Image.fromarray(slide_canvas)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        with col_img:
            st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
            st.subheader("Reconstructed specimen")
            st.image(slide_canvas, caption="", use_container_width=True)
            
            # Download Button
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
