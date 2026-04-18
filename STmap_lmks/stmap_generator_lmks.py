import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def RGB2YUV(RGBimg):
    transformation_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    img_rgb_reshaped = RGBimg.reshape(-1, 3).astype(np.float32)
    img_yuv_reshaped = img_rgb_reshaped @ transformation_matrix.T
    img_yuv_reshaped[:, 1:] += 128.0
    img_yuv = img_yuv_reshaped.reshape(RGBimg.shape)
    return np.clip(img_yuv, 0, 255).astype(np.uint8)

def get_frames(video_path):
    vid_obj = cv2.VideoCapture(video_path)
    total_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞 Total frames in video: {total_frames}")

    success, frame = vid_obj.read()
    frames = []

    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        success, frame = vid_obj.read()

    vid_obj.release()
    print(f"📸 Frames extracted: {len(frames)}")
    return np.asarray(frames)

def load_openface_landmarks(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # ← 이 줄이 핵심!

    all_lmks = []
    for _, row in df.iterrows():
        lm_x = np.array([row[f'x_{i}'] for i in range(68)])
        lm_y = np.array([row[f'y_{i}'] for i in range(68)])
        all_lmks.append(list(zip(lm_x, lm_y)))

    return all_lmks
def get_group_polygon_mean_color(frame, landmarks, group_indices):
    poly_points = np.array([landmarks[i] for i in group_indices], dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_points], 255)
    masked_pixels = frame[mask == 255]
    if len(masked_pixels) == 0:
        return np.array([0, 0, 0])
    return np.mean(masked_pixels, axis=0)

def STmap_from_lmk_polygons(frames, openface_lmks, lmk_groups):
    stmap_rows = []

    for group in lmk_groups:
        row = []
        for frame, lmks in zip(frames, openface_lmks):
            color = get_group_polygon_mean_color(frame, lmks, group)
            row.append(color)
        stmap_rows.append(row)

    stmap = np.array(stmap_rows, dtype=np.uint8)  # shape: (H, W, 3)
    return stmap

def save_STmap(stmap, save_path, filename, convert_to_bgr=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if convert_to_bgr:
        stmap = cv2.cvtColor(stmap, cv2.COLOR_RGB2BGR)
    save_file = os.path.join(save_path, filename)
    cv2.imwrite(save_file, stmap)

def main(raw_video_path, openface_lmk_csv_path):
    raw_frames = get_frames(raw_video_path)
    openface_lmks = load_openface_landmarks(openface_lmk_csv_path)

    if len(raw_frames) != len(openface_lmks):
        print("❌ 프레임 수와 랜드마크 수가 일치하지 않습니다.")
        return

    lmk_groups = [
        [8, 9, 10],
        [7, 8, 10, 11],
        [6, 7, 11, 12],  # 눈 주변
        [5, 6, 12, 13],          # 입꼬리
        [4, 5, 13, 14],          # 코/입 중앙
        [3, 4, 14, 15],
        [2, 3, 15, 16],
        [1, 2, 16, 17],
    ]

    stmap_rgb = STmap_from_lmk_polygons(raw_frames, openface_lmks, lmk_groups)
    save_dir = './output_stmaps'
    save_STmap(stmap_rgb, save_dir, 'STmap_RGB.png', convert_to_bgr=False)

    plt.figure(figsize=(10, 5))
    plt.imshow(stmap_rgb)
    plt.title('STmap from Landmark Polygons')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    raw_video_path = '/path/to/video.avi'
    openface_lmk_csv_path = '/path/to/openface_output.csv'  # OpenFace landmarks csv
    main(raw_video_path, openface_lmk_csv_path)
