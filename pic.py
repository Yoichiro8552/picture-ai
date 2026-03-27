import mediapipe as mp
import os
import cv2
import numpy as np
from datetime import datetime

# =========================
# 設定
# =========================
INPUT_PATH_1 = "person_size_tool/input/image1.jpg"
INPUT_PATH_2 = "person_size_tool/input/image2.jpg"

OUTPUT_PATH_1 = "person_size_tool/output/result1.jpg"
OUTPUT_PATH_2 = "person_size_tool/output/result2.jpg"

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

VIDEO_PATH_1 = "person_size_tool/input/video1.mp4"
VIDEO_PATH_2 = "person_size_tool/input/video2.mp4"

OUTPUT_VIDEO_PATH_1 = "person_size_tool/output/result_video1.mp4"
OUTPUT_VIDEO_PATH_2 = "person_size_tool/output/result_video2.mp4"

VIDEO_SAMPLE_STEP = 5

# =========================
# 画像読み込み
# =========================
def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {path}")

    return image


# =========================
# 人物情報取得
# =========================
def get_person_info(image, pose):
    h, w, _ = image.shape

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        raise ValueError("人物を検出できませんでした。")

    landmarks = result.pose_landmarks.landmark
    mp_pose = mp.solutions.pose

    def get_xy(landmark_enum, min_visibility=0.3):
        lm = landmarks[landmark_enum.value]
        if lm.visibility < min_visibility:
            return None
        x = int(lm.x * w)
        y = int(lm.y * h)
        return x, y

    # 上端候補（顔周り）
    top_candidates = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
    ]

    # 下端候補（足周り）
    bottom_candidates = [
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    ]

    # 中心候補（体幹寄り）
    center_candidates = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]

    top_ys = []
    bottom_ys = []
    center_xs = []

    for lm_enum in top_candidates:
        pt = get_xy(lm_enum)
        if pt is not None:
            _, y = pt
            top_ys.append(y)

    for lm_enum in bottom_candidates:
        pt = get_xy(lm_enum)
        if pt is not None:
            _, y = pt
            bottom_ys.append(y)

    for lm_enum in center_candidates:
        pt = get_xy(lm_enum)
        if pt is not None:
            x, _ = pt
            center_xs.append(x)

    if len(top_ys) == 0:
        raise ValueError("上端候補のランドマークが取得できませんでした。")

    if len(bottom_ys) == 0:
        raise ValueError("下端候補のランドマークが取得できませんでした。")

    if len(center_xs) == 0:
        raise ValueError("中心候補のランドマークが取得できませんでした。")

    top_y = max(0, min(top_ys))
    bottom_y = min(h - 1, max(bottom_ys))
    center_x = int(np.mean(center_xs))
    person_height = bottom_y - top_y

    if person_height <= 0:
        raise ValueError("人物高さを正しく計算できませんでした。人物が写真内に収まっていない可能性が高いです。")

    return {
        "top_y": top_y,
        "bottom_y": bottom_y,
        "center_x": center_x,
        "height": person_height
    }

def get_person_info_by_segmentation(image, pose, mask_threshold=0.35):
    h, w, _ = image.shape

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.segmentation_mask is None:
        raise ValueError("セグメンテーションマスクを取得できませんでした。")

    mask = result.segmentation_mask
    binary_mask = (mask > mask_threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("人物領域を検出できませんでした。")

    top_y = int(np.min(ys))
    bottom_y = int(np.max(ys))
    left_x = int(np.min(xs))
    right_x = int(np.max(xs))

    height = bottom_y - top_y
    width = right_x - left_x
    center_x = (left_x + right_x) // 2

    if height <= 0:
        raise ValueError("輪郭ベースの人物高さを正しく計算できませんでした。")

    return {
        "top_y": top_y,
        "bottom_y": bottom_y,
        "left_x": left_x,
        "right_x": right_x,
        "center_x": center_x,
        "height": height,
        "width": width,
        "mask": binary_mask
    }

def apply_person_focus_background(image, mask, background_color=(240, 240, 240)):
    """
    人物領域は元画像を残し、背景は指定色で塗る
    """
    if len(mask.shape) != 2:
        raise ValueError("mask は2次元配列である必要があります。")

    h, w = image.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        raise ValueError("image と mask のサイズが一致していません。")

    background = np.full((h, w, 3), background_color, dtype=np.uint8)
    mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.uint8)

    focused = np.where(mask_3ch == 1, image, background)
    return focused

# =========================
# 倍率計算
# =========================
def calculate_scale(height_a, height_b):
    if height_b == 0:
        raise ValueError("height_b が 0 です。倍率を計算できません。")
    return height_a / height_b


# =========================
# リサイズ
# =========================
def resize_image(image, scale):
    h, w = image.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized

def scale_person_info(info, scale):
    scaled = {
        "top_y": int(info["top_y"] * scale),
        "bottom_y": int(info["bottom_y"] * scale),
        "left_x": int(info["left_x"] * scale) if "left_x" in info else None,
        "right_x": int(info["right_x"] * scale) if "right_x" in info else None,
        "center_x": int(info["center_x"] * scale),
        "height": int(info["height"] * scale),
        "width": int(info["width"] * scale) if "width" in info else None
    }

    if "mask" in info:
        original_mask = info["mask"]
        h, w = original_mask.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        scaled_mask = cv2.resize(
            original_mask.astype(np.uint8),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )
        scaled["mask"] = scaled_mask

    return scaled

# =========================
# キャンバスに合わせる
# =========================
def fit_to_canvas(image, target_width, target_height):
    """
    画像を target_width x target_height の黒キャンバス中央に配置する。
    大きすぎる場合は中央トリミング、小さい場合は余白をつける。
    """
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    img_h, img_w = image.shape[:2]

    # 元画像から切り出す範囲
    src_x1 = max(0, (img_w - target_width) // 2)
    src_y1 = max(0, (img_h - target_height) // 2)
    src_x2 = min(img_w, src_x1 + target_width)
    src_y2 = min(img_h, src_y1 + target_height)

    cropped = image[src_y1:src_y2, src_x1:src_x2]

    crop_h, crop_w = cropped.shape[:2]

    # キャンバスへ貼る位置
    dst_x1 = max(0, (target_width - crop_w) // 2)
    dst_y1 = max(0, (target_height - crop_h) // 2)
    dst_x2 = dst_x1 + crop_w
    dst_y2 = dst_y1 + crop_h

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = cropped
    return canvas

def place_person_on_canvas(
    image,
    info,
    target_width,
    target_height,
    target_center_x,
    target_bottom_y,
    background_color=(255, 255, 255)
):
    """
    人物の center_x と bottom_y が、
    target_center_x, target_bottom_y に来るように画像をキャンバスへ配置する。
    はみ出す部分は自動で切り取る。
    """

    canvas = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)

    img_h, img_w = image.shape[:2]

    shift_x = target_center_x - info["center_x"]
    shift_y = target_bottom_y - info["bottom_y"]

    # キャンバス上で画像を置く位置
    dst_x1 = shift_x
    dst_y1 = shift_y
    dst_x2 = shift_x + img_w
    dst_y2 = shift_y + img_h

    # 元画像側の切り出し範囲
    src_x1 = 0
    src_y1 = 0
    src_x2 = img_w
    src_y2 = img_h

    # 左にはみ出した場合
    if dst_x1 < 0:
        src_x1 = -dst_x1
        dst_x1 = 0

    # 上にはみ出した場合
    if dst_y1 < 0:
        src_y1 = -dst_y1
        dst_y1 = 0

    # 右にはみ出した場合
    if dst_x2 > target_width:
        src_x2 = img_w - (dst_x2 - target_width)
        dst_x2 = target_width

    # 下にはみ出した場合
    if dst_y2 > target_height:
        src_y2 = img_h - (dst_y2 - target_height)
        dst_y2 = target_height

    # 貼り付け可能領域がない場合
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        raise ValueError("画像をキャンバスに配置できませんでした。")

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    return canvas

def combine_images_side_by_side(left_image, right_image, gap=20, background_color=(255, 255, 255)):
    left_h, left_w = left_image.shape[:2]
    right_h, right_w = right_image.shape[:2]

    output_height = max(left_h, right_h)
    output_width = left_w + gap + right_w

    canvas = np.full((output_height, output_width, 3), background_color, dtype=np.uint8)

    # 左画像を配置
    left_y = (output_height - left_h) // 2
    canvas[left_y:left_y + left_h, 0:left_w] = left_image

    # 右画像を配置
    right_y = (output_height - right_h) // 2
    start_x = left_w + gap
    canvas[right_y:right_y + right_h, start_x:start_x + right_w] = right_image

    return canvas
# =========================
# 画像保存
# =========================
def save_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ext = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(ext, image)

    if not success:
        raise IOError(f"画像をエンコードできませんでした: {path}")

    encoded.tofile(path)

def save_mask_image(path, mask):
    mask_img = (mask * 255).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    save_image(path, mask_bgr)

# =========================
# デバッグ用：ランドマーク描画
# =========================
def draw_person_info(image, info):
    debug_img = image.copy()
    h, w = debug_img.shape[:2]

    # 線の太さ
    outer_thickness = 15   # 黒い縁
    inner_thickness = 6   # 色付き本線

    top_y = info["top_y"]
    bottom_y = info["bottom_y"]
    center_x = info["center_x"]

    # ===== 上端ライン（緑）=====
    cv2.line(debug_img, (0, top_y), (w - 1, top_y), (0, 0, 0), outer_thickness)
    cv2.line(debug_img, (0, top_y), (w - 1, top_y), (0, 255, 0), inner_thickness)

    # ===== 下端ライン（赤）=====
    cv2.line(debug_img, (0, bottom_y), (w - 1, bottom_y), (0, 0, 0), outer_thickness)
    cv2.line(debug_img, (0, bottom_y), (w - 1, bottom_y), (0, 0, 255), inner_thickness)

    # ===== 中心線（青）=====
    cv2.line(debug_img, (center_x, 0), (center_x, h - 1), (0, 0, 0), outer_thickness)
    cv2.line(debug_img, (center_x, 0), (center_x, h - 1), (255, 0, 0), inner_thickness)


    return debug_img

def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count
    }

def estimate_video_max_height(video_path, pose, sample_step=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {video_path}")

    heights = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % sample_step == 0:
            try:
                info_seg = get_person_info_by_segmentation(frame, pose)
                heights.append(info_seg["height"])
            except Exception:
                pass

        frame_index += 1

    cap.release()

    if len(heights) == 0:
        raise ValueError(f"動画から人物高さを取得できませんでした: {video_path}")

    return float(np.max(heights))

def process_video(
    input_video_path,
    output_video_path,
    pose,
    base_scale,
    target_width,
    target_height,
    target_center_x,
    target_bottom_y,
    background_color=(255, 255, 255),
    smooth_alpha=0.8,
    dynamic_scale=False,
    reference_height=None,
    min_scale=0.5,
    max_scale=2.0
):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {input_video_path}")

    props = get_video_properties(cap)

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        props["fps"],
        (target_width, target_height)
    )

    processed_count = 0
    reused_count = 0
    blank_count = 0

    last_valid_frame = None
    prev_center_x = None
    prev_bottom_y = None
    prev_scale = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            info_seg = get_person_info_by_segmentation(frame, pose)

            current_scale = base_scale

            # フレームごとに倍率を変える場合
            if dynamic_scale:
                if reference_height is None:
                    raise ValueError("dynamic_scale=True のときは reference_height が必要です。")

                current_height = info_seg["height"]
                if current_height <= 0:
                    raise ValueError("現在フレームの人物高さが不正です。")

                current_scale = reference_height / current_height

                # 極端な倍率を防ぐ
                current_scale = max(min_scale, min(max_scale, current_scale))

                # 倍率もなめらかにする
                if prev_scale is not None:
                    current_scale = smooth_alpha * prev_scale + (1.0 - smooth_alpha) * current_scale

                prev_scale = current_scale

            if current_scale != 1.0:
                frame = resize_image(frame, current_scale)
                info_seg = scale_person_info(info_seg, current_scale)

            current_center_x = info_seg["center_x"]
            current_bottom_y = info_seg["bottom_y"]

            # center_x をなめらかにする
            if prev_center_x is not None:
                smoothed_center_x = int(smooth_alpha * prev_center_x + (1.0 - smooth_alpha) * current_center_x)
            else:
                smoothed_center_x = current_center_x

            # bottom_y をなめらかにする
            if prev_bottom_y is not None:
                smoothed_bottom_y = int(smooth_alpha * prev_bottom_y + (1.0 - smooth_alpha) * current_bottom_y)
            else:
                smoothed_bottom_y = current_bottom_y

            prev_center_x = smoothed_center_x
            prev_bottom_y = smoothed_bottom_y

            smoothed_info = info_seg.copy()
            smoothed_info["center_x"] = smoothed_center_x
            smoothed_info["bottom_y"] = smoothed_bottom_y

            out_frame = place_person_on_canvas(
                frame,
                smoothed_info,
                target_width,
                target_height,
                target_center_x,
                target_bottom_y,
                background_color=background_color
            )

            writer.write(out_frame)
            last_valid_frame = out_frame.copy()
            processed_count += 1

        except Exception:
            if last_valid_frame is not None:
                writer.write(last_valid_frame)
                reused_count += 1
            else:
                blank = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
                writer.write(blank)
                blank_count += 1

    cap.release()
    writer.release()

    print(f"{input_video_path} の処理完了")
    print(f"  正常処理フレーム数: {processed_count}")
    print(f"  再利用フレーム数: {reused_count}")
    print(f"  白フレーム数: {blank_count}")
    print(f"  保存先: {output_video_path}")

def process_image_pair(input_path_1, input_path_2, output_path_1, output_path_2):
    print("画像処理を開始します...")

    img1 = load_image(input_path_1)
    img2 = load_image(input_path_2)

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        # Poseベース
        info1 = get_person_info(img1, pose)
        info2 = get_person_info(img2, pose)

        # 輪郭ベース
        info1_seg = get_person_info_by_segmentation(img1, pose)
        info2_seg = get_person_info_by_segmentation(img2, pose)

    print("=== Poseベース ===")
    print(f"image1 の人物高さ: {info1['height']} px")
    print(f"image2 の人物高さ: {info2['height']} px")

    print("=== 輪郭ベース ===")
    print(f"image1 の人物高さ: {info1_seg['height']} px")
    print(f"image2 の人物高さ: {info2_seg['height']} px")

    scale = calculate_scale(info1_seg["height"], info2_seg["height"])
    print(f"image2 にかける倍率: {scale:.4f}")

    if scale < 0.5 or scale > 2.0:
        print("注意: 倍率がかなり大きい/小さいです。人物検出結果を確認してください。")

    save_mask_image("output/mask1.jpg", info1_seg["mask"])
    save_mask_image("output/mask2.jpg", info2_seg["mask"])

    img2_scaled = resize_image(img2, scale)
    info2_seg_scaled = scale_person_info(info2_seg, scale)

    target_center_x = TARGET_WIDTH // 2
    target_bottom_y = int(TARGET_HEIGHT * 0.85)

    out1 = place_person_on_canvas(
        img1,
        info1_seg,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        target_center_x,
        target_bottom_y,
        background_color=(255, 255, 255)
    )

    out2 = place_person_on_canvas(
        img2_scaled,
        info2_seg_scaled,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        target_center_x,
        target_bottom_y,
        background_color=(255, 255, 255)
    )

    save_image(output_path_1, out1)
    save_image(output_path_2, out2)

    # デバッグ画像
    debug1 = draw_person_info(img1, info1)
    debug2 = draw_person_info(img2, info2)
    save_image("output/debug_image1.jpg", debug1)
    save_image("output/debug_image2.jpg", debug2)

    placed_debug1 = draw_person_info(out1, {
        "top_y": target_bottom_y - info1_seg["height"],
        "bottom_y": target_bottom_y,
        "center_x": target_center_x,
        "height": info1_seg["height"]
    })

    placed_debug2 = draw_person_info(out2, {
        "top_y": target_bottom_y - info2_seg_scaled["height"],
        "bottom_y": target_bottom_y,
        "center_x": target_center_x,
        "height": info2_seg_scaled["height"]
    })

    save_image("output/placed_debug_image1.jpg", placed_debug1)
    save_image("output/placed_debug_image2.jpg", placed_debug2)

    print("画像処理が完了しました。")
    print(f"保存先: {output_path_1}")
    print(f"保存先: {output_path_2}")

def process_image_pair_combined(input_path_1, input_path_2, output_path, debug_output_path):
    print("画像比較処理を開始します...")

    img1 = load_image(input_path_1)
    img2 = load_image(input_path_2)

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        info1_seg = get_person_info_by_segmentation(img1, pose)
        info2_seg = get_person_info_by_segmentation(img2, pose)

    print(f"image1 の人物高さ: {info1_seg['height']} px")
    print(f"image2 の人物高さ: {info2_seg['height']} px")

    # 共通基準高さ
    reference_height = max(info1_seg["height"], info2_seg["height"])
    print(f"共通基準高さ: {reference_height} px")

    # 両方の倍率を計算
    scale1 = reference_height / info1_seg["height"]
    scale2 = reference_height / info2_seg["height"]

    print(f"image1 にかける倍率: {scale1:.4f}")
    print(f"image2 にかける倍率: {scale2:.4f}")

    # 両方をリサイズ
    img1_scaled = resize_image(img1, scale1)
    img2_scaled = resize_image(img2, scale2)

    info1_seg_scaled = scale_person_info(info1_seg, scale1)
    info2_seg_scaled = scale_person_info(info2_seg, scale2)

    # 各画像を同じ縦長キャンバスに配置
    target_center_x = TARGET_WIDTH // 2
    target_bottom_y = int(TARGET_HEIGHT * 0.85)

    out1 = place_person_on_canvas(
        img1_scaled,
        info1_seg_scaled,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        target_center_x,
        target_bottom_y,
        background_color=(255, 255, 255)
    )

    out2 = place_person_on_canvas(
        img2_scaled,
        info2_seg_scaled,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        target_center_x,
        target_bottom_y,
        background_color=(255, 255, 255)
    )

    # 本番用：線なし画像を左右結合
    combined_result = combine_images_side_by_side(
        out1,
        out2,
        gap=20,
        background_color=(255, 255, 255)
    )

    save_image(output_path, combined_result)

    # 確認用：線あり画像を作る
    placed_debug1 = draw_person_info(out1, {
        "top_y": target_bottom_y - info1_seg_scaled["height"],
        "bottom_y": target_bottom_y,
        "center_x": target_center_x,
        "height": info1_seg_scaled["height"]
    })

    placed_debug2 = draw_person_info(out2, {
        "top_y": target_bottom_y - info2_seg_scaled["height"],
        "bottom_y": target_bottom_y,
        "center_x": target_center_x,
        "height": info2_seg_scaled["height"]
    })

    combined_debug_result = combine_images_side_by_side(
        placed_debug1,
        placed_debug2,
        gap=20,
        background_color=(255, 255, 255)
    )

    save_image(debug_output_path, combined_debug_result)

    # 確認用マスクも保存
    save_mask_image("output/mask1.jpg", info1_seg["mask"])
    save_mask_image("output/mask2.jpg", info2_seg["mask"])

    print("画像比較処理が完了しました。")
    print(f"本番用保存先: {output_path}")
    print(f"確認用保存先: {debug_output_path}")

def process_video_pair(input_path_1, input_path_2, output_path_1, output_path_2):
    print("動画処理を開始します...")

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        # 基準は video1 の最大人物高さ
        max_height_1 = estimate_video_max_height(
            input_path_1, pose, sample_step=VIDEO_SAMPLE_STEP
        )
        max_height_2 = estimate_video_max_height(
            input_path_2, pose, sample_step=VIDEO_SAMPLE_STEP
        )

        print(f"video1 の最大人物高さ: {max_height_1:.2f} px")
        print(f"video2 の最大人物高さ: {max_height_2:.2f} px")

        target_center_x = TARGET_WIDTH // 2
        target_bottom_y = int(TARGET_HEIGHT * 0.85)

        # 動画1も、video1 の最大人物高さに合わせて各フレームで倍率を変える
        process_video(
            input_path_1,
            output_path_1,
            pose,
            base_scale=1.0,
            target_width=TARGET_WIDTH,
            target_height=TARGET_HEIGHT,
            target_center_x=target_center_x,
            target_bottom_y=target_bottom_y,
            background_color=(255, 255, 255),
            smooth_alpha=0.8,
            dynamic_scale=True,
            reference_height=max_height_1,
            min_scale=0.1,
            max_scale=9.0
        )

        # 動画2も、同じ基準に合わせて各フレームで倍率を変える
        process_video(
            input_path_2,
            output_path_2,
            pose,
            base_scale=1.0,
            target_width=TARGET_WIDTH,
            target_height=TARGET_HEIGHT,
            target_center_x=target_center_x,
            target_bottom_y=target_bottom_y,
            background_color=(255, 255, 255),
            smooth_alpha=0.8,
            dynamic_scale=True,
            reference_height=max_height_1,
            min_scale=0.1,
            max_scale=9.0
        )

    print("動画処理が完了しました。")
    print(f"保存先: {output_path_1}")
    print(f"保存先: {output_path_2}")

def make_debug_frame(frame, info, target_center_x, target_bottom_y):
    return draw_person_info(frame, {
        "top_y": target_bottom_y - info["height"],
        "bottom_y": target_bottom_y,
        "center_x": target_center_x,
        "height": info["height"]
    })

def process_video_pair_combined(input_path_1, input_path_2, output_path, debug_output_path):
    print("動画比較処理を開始します...")

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        # まず両動画の最大人物高さを求める
        max_height_1 = estimate_video_max_height(
            input_path_1, pose, sample_step=VIDEO_SAMPLE_STEP
        )
        max_height_2 = estimate_video_max_height(
            input_path_2, pose, sample_step=VIDEO_SAMPLE_STEP
        )

        reference_height = max(max_height_1, max_height_2)

        print(f"video1 の最大人物高さ: {max_height_1:.2f} px")
        print(f"video2 の最大人物高さ: {max_height_2:.2f} px")
        print(f"共通基準高さ: {reference_height:.2f} px")

        cap1 = cv2.VideoCapture(input_path_1)
        cap2 = cv2.VideoCapture(input_path_2)

        if not cap1.isOpened():
            raise FileNotFoundError(f"動画を開けませんでした: {input_path_1}")
        if not cap2.isOpened():
            raise FileNotFoundError(f"動画を開けませんでした: {input_path_2}")

        props1 = get_video_properties(cap1)
        props2 = get_video_properties(cap2)

        fps = min(props1["fps"], props2["fps"])
        if fps <= 0:
            fps = 30.0

        gap = 20
        combined_width = TARGET_WIDTH * 2 + gap
        combined_height = TARGET_HEIGHT

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (combined_width, combined_height)
        )
        debug_writer = cv2.VideoWriter(
            debug_output_path,
            fourcc,
            fps,
            (combined_width, combined_height)
        )

        target_center_x = TARGET_WIDTH // 2
        target_bottom_y = int(TARGET_HEIGHT * 0.85)

        smooth_alpha = 0.8
        min_scale = 0.5
        max_scale = 2.0

        prev_scale1 = None
        prev_scale2 = None
        prev_center_x1 = None
        prev_bottom_y1 = None
        prev_center_x2 = None
        prev_bottom_y2 = None

        last_valid_combined = None
        last_valid_combined_debug = None

        processed_count = 0
        reused_count = 0
        blank_count = 0

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            # どちらかが終わったら終了
            if not ret1 or not ret2:
                break

            try:
                info1_seg = get_person_info_by_segmentation(frame1, pose)
                info2_seg = get_person_info_by_segmentation(frame2, pose)

                focused_frame1 = apply_person_focus_background(
                frame1,
                info1_seg["mask"],
                background_color=(240, 240, 240)
                )

                focused_frame2 = apply_person_focus_background(
                frame2,
                info2_seg["mask"],
                background_color=(240, 240, 240)
                )

                current_height_1 = info1_seg["height"]
                current_height_2 = info2_seg["height"]

                if current_height_1 <= 0 or current_height_2 <= 0:
                    raise ValueError("人物高さが不正です。")

                # 各フレームで倍率を決める
                scale1 = reference_height / current_height_1
                scale2 = reference_height / current_height_2

                # 極端な倍率を防ぐ
                scale1 = max(min_scale, min(max_scale, scale1))
                scale2 = max(min_scale, min(max_scale, scale2))

                # 倍率をなめらかにする
                if prev_scale1 is not None:
                    scale1 = smooth_alpha * prev_scale1 + (1.0 - smooth_alpha) * scale1
                if prev_scale2 is not None:
                    scale2 = smooth_alpha * prev_scale2 + (1.0 - smooth_alpha) * scale2

                prev_scale1 = scale1
                prev_scale2 = scale2

                # 拡大縮小
                frame1_scaled = resize_image(focused_frame1, scale1)
                frame2_scaled = resize_image(focused_frame2, scale2)

                info1_scaled = scale_person_info(info1_seg, scale1)
                info2_scaled = scale_person_info(info2_seg, scale2)

                # center_x / bottom_y をなめらかにする
                current_center_x1 = info1_scaled["center_x"]
                current_bottom_y1 = info1_scaled["bottom_y"]
                current_center_x2 = info2_scaled["center_x"]
                current_bottom_y2 = info2_scaled["bottom_y"]

                if prev_center_x1 is not None:
                    smoothed_center_x1 = int(smooth_alpha * prev_center_x1 + (1.0 - smooth_alpha) * current_center_x1)
                else:
                    smoothed_center_x1 = current_center_x1

                if prev_bottom_y1 is not None:
                    smoothed_bottom_y1 = int(smooth_alpha * prev_bottom_y1 + (1.0 - smooth_alpha) * current_bottom_y1)
                else:
                    smoothed_bottom_y1 = current_bottom_y1

                if prev_center_x2 is not None:
                    smoothed_center_x2 = int(smooth_alpha * prev_center_x2 + (1.0 - smooth_alpha) * current_center_x2)
                else:
                    smoothed_center_x2 = current_center_x2

                if prev_bottom_y2 is not None:
                    smoothed_bottom_y2 = int(smooth_alpha * prev_bottom_y2 + (1.0 - smooth_alpha) * current_bottom_y2)
                else:
                    smoothed_bottom_y2 = current_bottom_y2

                prev_center_x1 = smoothed_center_x1
                prev_bottom_y1 = smoothed_bottom_y1
                prev_center_x2 = smoothed_center_x2
                prev_bottom_y2 = smoothed_bottom_y2

                smoothed_info1 = info1_scaled.copy()
                smoothed_info2 = info2_scaled.copy()

                smoothed_info1["center_x"] = smoothed_center_x1
                smoothed_info1["bottom_y"] = smoothed_bottom_y1
                smoothed_info2["center_x"] = smoothed_center_x2
                smoothed_info2["bottom_y"] = smoothed_bottom_y2

                # 左右の完成フレーム
                out1 = place_person_on_canvas(
                    frame1_scaled,
                    smoothed_info1,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                    target_center_x,
                    target_bottom_y,
                    background_color=(255, 255, 255)
                )

                out2 = place_person_on_canvas(
                    frame2_scaled,
                    smoothed_info2,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                    target_center_x,
                    target_bottom_y,
                    background_color=(255, 255, 255)
                )

                # 本番用横結合
                combined_frame = combine_images_side_by_side(
                    out1, out2, gap=gap, background_color=(255, 255, 255)
                )

                # 確認用フレーム
                debug1 = make_debug_frame(out1, smoothed_info1, target_center_x, target_bottom_y)
                debug2 = make_debug_frame(out2, smoothed_info2, target_center_x, target_bottom_y)

                combined_debug_frame = combine_images_side_by_side(
                    debug1, debug2, gap=gap, background_color=(255, 255, 255)
                )

                writer.write(combined_frame)
                debug_writer.write(combined_debug_frame)

                last_valid_combined = combined_frame.copy()
                last_valid_combined_debug = combined_debug_frame.copy()

                processed_count += 1

            except Exception:
                if last_valid_combined is not None and last_valid_combined_debug is not None:
                    writer.write(last_valid_combined)
                    debug_writer.write(last_valid_combined_debug)
                    reused_count += 1
                else:
                    blank = np.full((combined_height, combined_width, 3), 255, dtype=np.uint8)
                    writer.write(blank)
                    debug_writer.write(blank)
                    blank_count += 1

        cap1.release()
        cap2.release()
        writer.release()
        debug_writer.release()

        print("動画比較処理が完了しました。")
        print(f"正常処理フレーム数: {processed_count}")
        print(f"再利用フレーム数: {reused_count}")
        print(f"白フレーム数: {blank_count}")
        print(f"本番用保存先: {output_path}")
        print(f"確認用保存先: {debug_output_path}")

def is_image_file(path):
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ext = os.path.splitext(path)[1].lower()
    return ext in image_exts

def is_video_file(path):
    video_exts = [".mp4", ".mov", ".avi", ".mkv", ".wmv"]
    ext = os.path.splitext(path)[1].lower()
    return ext in video_exts

def make_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_output_paths(output_dir, mode):
    timestamp = make_timestamp()

    if mode == "image":
        main_output = os.path.join(output_dir, f"{timestamp}_combined_result.jpg")
        debug_output = os.path.join(output_dir, f"{timestamp}_combined_debug_result.jpg")
    elif mode == "video":
        main_output = os.path.join(output_dir, f"{timestamp}_combined_result.mp4")
        debug_output = os.path.join(output_dir, f"{timestamp}_combined_debug_result.mp4")
    else:
        raise ValueError(f"不明な mode です: {mode}")

    return main_output, debug_output

def process_pair(input_path_1, input_path_2, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    is_img_1 = is_image_file(input_path_1)
    is_img_2 = is_image_file(input_path_2)
    is_vid_1 = is_video_file(input_path_1)
    is_vid_2 = is_video_file(input_path_2)

    # 画像2枚の場合
    # 画像2枚の場合
    if is_img_1 and is_img_2:
        output_path, debug_output_path = make_output_paths(output_dir, "image")

        print("入力は画像2枚と判断しました。")
        process_image_pair_combined(
            input_path_1,
            input_path_2,
            output_path,
            debug_output_path
        )

        return {
            "mode": "image",
            "main_output": output_path,
            "debug_output": debug_output_path
        }

    # 動画2本の場合
    elif is_vid_1 and is_vid_2:
        output_path, debug_output_path = make_output_paths(output_dir, "video")

        print("入力は動画2本と判断しました。")
        process_video_pair_combined(
            input_path_1,
            input_path_2,
            output_path,
            debug_output_path
        )

        return {
            "mode": "video",
            "main_output": output_path,
            "debug_output": debug_output_path
        }

    # 画像と動画が混ざっている場合
    else:
        raise ValueError(
            "入力ファイルの組み合わせが不正です。"
            "画像2枚または動画2本を選んでください。"
        )

def get_input_files(input_dir="input"):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"入力フォルダが見つかりません: {input_dir}")

    files = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path):
            files.append(path)

    files.sort()
    return files

def get_two_input_files(input_dir="input"):
    files = get_input_files(input_dir)

    if len(files) != 2:
        raise ValueError(
            f"{input_dir} フォルダにはちょうど2つのファイルを入れてください。"
            f"現在のファイル数: {len(files)}"
        )

    return files[0], files[1]


# =========================
# メイン処理
# =========================
def main():
    input_path_1, input_path_2 = get_two_input_files("input")

    print(f"入力ファイル1: {input_path_1}")
    print(f"入力ファイル2: {input_path_2}")

    result = process_pair(
        input_path_1,
        input_path_2,
        output_dir="output"
    )

    print(f"処理モード: {result['mode']}")
    print(f"本番用出力: {result['main_output']}")
    print(f"確認用出力: {result['debug_output']}")

def run_app_process(file1, file2):
    if file1 is None or file2 is None:
        return "ファイルを2つ選んでください。", None, None

    try:
        # Gradio から渡される値が文字列パスかファイルオブジェクトかの違いに対応
        path1 = file1 if isinstance(file1, str) else file1.name
        path2 = file2 if isinstance(file2, str) else file2.name

        result = process_pair(path1, path2, output_dir="output")

        mode = result["mode"]
        main_output = result["main_output"]
        debug_output = result["debug_output"]

        if mode == "image":
            message = "画像比較処理が完了しました。"
        elif mode == "video":
            message = "動画比較処理が完了しました。"
        else:
            message = "処理が完了しました。"

        return message, main_output, debug_output

    except Exception as e:
        return f"エラーが発生しました: {str(e)}", None, None

def launch_gradio_app():
    import gradio as gr

    with gr.Blocks(title="人物サイズ比較ツール") as demo:
        gr.Markdown("# 人物サイズ比較ツール")
        gr.Markdown(
            "画像2枚または動画2本を入れると、人物サイズをそろえて左右比較結果を出力します。"
        )
        gr.Markdown(
            "【使い方】\n"
            "1. ファイルを2つ選びます。\n"
            "2. 「処理開始」を押します。\n"
            "3. 比較結果と確認結果を受け取ります。"
        )
        gr.Markdown(
            "【注意】\n"
            "- 縦向きの画像・動画を想定しています。\n"
            "- 画像2枚または動画2本を選んでください。\n"
            "- 画像と動画をまぜて入れることはできません。"
        )

        with gr.Row():
            file1 = gr.File(label="ファイル1")
            file2 = gr.File(label="ファイル2")

        run_button = gr.Button("処理開始")

        status_text = gr.Textbox(label="状態", interactive=False)
        output_main_file = gr.File(label="比較結果（本番用）")
        output_debug_file = gr.File(label="確認結果（線あり）")

        run_button.click(
            fn=run_app_process,
            inputs=[file1, file2],
            outputs=[status_text, output_main_file, output_debug_file]
        )

    demo.launch(inbrowser=True)

if __name__ == "__main__":
    print("====================================")
    print("人物サイズ比較ツールを起動しています。")
    print("ブラウザが自動で開きます。少し待ってください。")
    print("この黒い画面を閉じると、アプリも終了します。")
    print("====================================")
    launch_gradio_app()