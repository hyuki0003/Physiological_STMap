import os
from tqdm import tqdm
import cv2
import torch
import stmap_generator_lmks as su  # 필요한 유틸 함수 포함되어 있어야 함

def find_avi_files_by_dir(directory):
    avi_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".avi"):
                file_path = os.path.join(root, file)
                subdir = os.path.relpath(root, directory)
                if subdir not in avi_files:
                    avi_files[subdir] = []
                avi_files[subdir].append(file_path)
    return avi_files

def initialize_cuda():
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA is available. Number of devices: {device_count}")
        else:
            print("CUDA is not available.")
    except Exception as e:
        print(f"Error initializing CUDA: {e}")

def process_video_for_stmap(raw_video_path, subdir, output_stmap_root_path, openface_csv_root_path):
    try:
        input_filename = os.path.splitext(os.path.basename(raw_video_path))[0]
        output_dir = os.path.join(output_stmap_root_path, subdir)
        os.makedirs(output_dir, exist_ok=True)

        raw_frames = su.get_frames(raw_video_path)
        if len(raw_frames) == 0:
            print(f"❌ 비디오에서 프레임을 찾을 수 없습니다: {raw_video_path}")
            return

        lmk_csv_path = os.path.join(openface_csv_root_path, subdir, "vid.csv")

        print(f"🔍 CSV 경로 확인: {lmk_csv_path}")
        if not os.path.exists(lmk_csv_path):
            print(f"❌ OpenFace CSV 파일이 없습니다: {lmk_csv_path}")
            return

        openface_lmks = su.load_openface_landmarks(lmk_csv_path)
        if len(openface_lmks) != len(raw_frames):
            print(f"❌ 프레임 수와 랜드마크 수 불일치: {raw_video_path}")
            return

        lmk_groups = [
            [8, 9, 10],
            [7, 8, 10, 11],
            [6, 7, 11, 12],
            [5, 6, 12, 13],
            [4, 5, 13, 14],
            [3, 4, 14, 15],
            [2, 3, 15, 16],
            [1, 2, 16, 17],
        ]

        stmap_rgb = su.STmap_from_lmk_polygons(raw_frames, openface_lmks, lmk_groups)
        stmap_yuv = su.RGB2YUV(stmap_rgb)
        su.save_STmap(stmap_rgb, output_dir, f'{input_filename}_stmap_rgb.png', convert_to_bgr=False)
        su.save_STmap(stmap_yuv, output_dir, f'{input_filename}_stmap_yuv.png', convert_to_bgr=False)
        print(f"✅ {subdir}/{input_filename} STmap 저장 완료")

    except Exception as e:
        print(f"❌ 비디오 {raw_video_path} 처리 실패: {e}")

def create_stmap_for_videos(raw_vid_dir_path, output_stmap_root_path, openface_csv_root_path):
    raw_files = find_avi_files_by_dir(raw_vid_dir_path)

    for subdir, files in tqdm(raw_files.items(), desc="Creating STmaps"):
        for file in files:
            tqdm.write(f"Processing Raw: {file}, Subdir: {subdir}")
            process_video_for_stmap(file, subdir, output_stmap_root_path, openface_csv_root_path)

if __name__ == "__main__":
    initialize_cuda()

    raw_vid_path = '/media/neuroai/T7/rPPG/UBFC-rPPG/DATASET_2'
    openface_csv_root = '/media/neuroai/T7/rPPG/OpenFace_lmks/UBFC'
    output_stmap_path = '/media/neuroai/T7/rPPG/STMap_lmks/UBFC'

    create_stmap_for_videos(raw_vid_path, output_stmap_path, openface_csv_root)
