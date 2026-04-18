import cv2
import numpy as np
import face_alignment
from scipy.interpolate import splrep, splev
import os
import matplotlib.pyplot as plt
import torch


def get_frames(video_path):
    vid_obj = cv2.VideoCapture(video_path)

    # ✅ 비디오의 총 프레임 개수 확인
    total_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞 Total frames in video: {total_frames}")  # 원래 영상의 총 프레임 개수 출력

    success, frame = vid_obj.read()
    frames = []
    count = 0  # 읽은 프레임 개수

    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        success, frame = vid_obj.read()
        count += 1  # 실제로 읽힌 프레임 개수 증가

    vid_obj.release()

    print(f"📸 Frames extracted: {count}")  # 실제 추출된 프레임 개수 확인
    return np.asarray(frames)


def get_landmarks(model, frames):
    lmks = []
    abnormal_indices = []

    for i, frame in enumerate(frames):
        lmk = model.get_landmarks(frame)
        if lmk is None:
            abnormal_indices.append(i)
        else:
            lmks.append(lmk[0].reshape(136))

    frame_indices = np.arange(len(frames))
    normal_indices = np.delete(frame_indices, abnormal_indices)
    lmks = np.asarray(lmks)

    interpolated_lmks = []
    for i in range(136):
        ith_lmks = lmks[:, i]
        if len(normal_indices) != len(ith_lmks):
            print(f"Length mismatch: normal_indices({len(normal_indices)}), ith_lmks({len(ith_lmks)})")
            continue
        spline_representation = splrep(normal_indices, ith_lmks)
        interpolated_lmk = splev(frame_indices, spline_representation)
        interpolated_lmks.append(interpolated_lmk)

    interpolated_lmks = np.array(interpolated_lmks).T

    print(f"🛠 Interpolated landmarks shape: {interpolated_lmks.shape}")

    return interpolated_lmks

def align_face(frames, lmks):
    aligned_face_list = []
    src_points = np.array([[0, 48], [128, 48], [64, 128]], dtype=np.float32)

    for i, frame in enumerate(frames):
        lmk = lmks[i].reshape(-1, 2)
        dst_points = np.array([lmk[1], lmk[15], lmk[8]], dtype=np.float32)
        M = cv2.getAffineTransform(dst_points, src_points)
        face_aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        aligned_face_list.append(face_aligned[:128, :128])

    print(f"📏 Aligned faces count: {len(aligned_face_list)}")

    return np.asarray(aligned_face_list)

def get_STvalues(frame):
    STvalues = []
    frame = frame[:, :]
    frame = np.flipud(frame)

    n_wROI = 1
    n_hROI = frame.shape[0]

    h, w, _ = frame.shape
    w_step = w // n_wROI
    h_step = h // n_hROI

    for w_index in range(n_wROI):
        for h_index in range(n_hROI):
            ROI = frame[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
            ROI_mean_value = np.nanmean(np.nanmean(ROI, axis=0), axis=0)
            STvalues.append(ROI_mean_value)

    return STvalues


def STmap(aligned_face_list):
    STMap = []
    for RGBframe in aligned_face_list:
        YUVframe = RGB2YUV(RGBframe)
        Value = get_STvalues(YUVframe)
        STMap.append(Value)

    STMap = np.array(STMap)
    nanmin = np.nanmin(STMap, axis=0, keepdims=True)
    nanmax = np.nanmax(STMap, axis=0, keepdims=True)
    normalized_STMap = 255 * (STMap - nanmin) / (1e-7 + nanmax - nanmin)
    transposed_STMap = np.swapaxes(normalized_STMap, 0, 1)
    int_STMap = np.rint(transposed_STMap)
    return np.uint8(int_STMap)


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


def YUV2RGB(YUVimg):
    inverse_transformation_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])

    img_yuv_reshaped = YUVimg.reshape(-1, 3).astype(np.float32)
    img_yuv_reshaped[:, 1:] -= 128.0
    img_rgb_reshaped = img_yuv_reshaped @ inverse_transformation_matrix.T
    img_rgb = img_rgb_reshaped.reshape(YUVimg.shape)
    return np.clip(img_rgb, 0, 255).astype(np.uint8)


def save_STmap(stmap, save_path, filename, convert_to_bgr=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if convert_to_bgr:
        stmap = cv2.cvtColor(stmap, cv2.COLOR_RGB2BGR)

    save_file = os.path.join(save_path, filename)
    cv2.imwrite(save_file, stmap)


def main(raw_video_path, diff_video_path):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False,
                                      device='cuda' if torch.cuda.is_available() else 'cpu')

    raw_frames = get_frames(raw_video_path)
    lmks = get_landmarks(fa, raw_frames)
    # diff_frames = get_frames(diff_video_path)

    aligned_faces = align_face(raw_frames, lmks)
    stmap_yuv = STmap(aligned_faces)
    stmap_rgb = YUV2RGB(stmap_yuv)

    save_STmap(stmap_yuv, '/home/neuroai/Projects/DSTMap/DSTMap/temp2', 'STmap_YUV.png')
    save_STmap(stmap_rgb, '/home/neuroai/Projects/DSTMap/DSTMap/temp2', 'STmap_RGB.png', convert_to_bgr=False)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(stmap_yuv, cmap='gray')
    plt.title('STmap YUV')

    plt.subplot(1, 2, 2)
    plt.imshow(stmap_rgb)
    plt.title('STmap RGB')

    plt.show()


if __name__ == "__main__":
    raw_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/vid.avi'  # 비디오 파일 경로를 설정하세요
    diff_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/DIFFVideos_v2/v2_BN_output_video.avi'
    main(raw_video_path, diff_video_path)


