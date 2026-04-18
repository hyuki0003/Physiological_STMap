import os
from tqdm import tqdm
import face_alignment
import torch
import cv2  # OpenCV를 임포트하여 BGR을 RGB로 변환
import stmap_generator as su  # STmap 유틸리티 함수들을 임포트


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


def process_video_for_stmap(raw_video_path, subdir, output_stmap_root_path):
    try:
        input_filename = os.path.splitext(os.path.basename(raw_video_path))[0]
        output_dir = os.path.join(output_stmap_root_path, subdir)
        os.makedirs(output_dir, exist_ok=True)

        raw_frames = su.get_frames(raw_video_path)
        if len(raw_frames) == 0:
            print(f"비디오에서 프레임을 찾을 수 없습니다: {raw_video_path}")
            return

        # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
        #                                   device='cuda' if torch.cuda.is_available() else 'cpu')
        # lmks = su.get_landmarks(fa, raw_frames)
        # if len(lmks) == 0:
        #     print(f"비디오에서 랜드마크를 찾을 수 없습니다: {raw_video_path}")
        #     return
        #
        # aligned_faces = su.align_face(raw_frames, lmks)

        stmap_yuv = su.STmap(raw_frames)
        stmap_rgb = su.YUV2RGB(stmap_yuv)
        su.save_STmap(stmap_yuv, output_dir, f'{input_filename}_stmap_yuv.png')
        su.save_STmap(stmap_rgb, output_dir, f'{input_filename}_stmap_rgb.png', convert_to_bgr=False)
        print(f"{subdir}의 {input_filename}에 대한 ST맵을 처리하고 저장했습니다.")

    except Exception as e:
        print(f"비디오 {raw_video_path} 처리 실패: {e}")


def create_stmap_for_videos(raw_vid_dir_path, output_stmap_root_path):

    # for UBFC_rPPG
    raw_files = find_avi_files_by_dir(raw_vid_dir_path)

    for subdir, files in tqdm(raw_files.items(), desc="Creating STmaps"):
        for file in files:
            tqdm.write(f"Processing Raw: {file}, Subdir: {subdir}")
            process_video_for_stmap(file, subdir, output_stmap_root_path)


if __name__ == "__main__":
    initialize_cuda()

    raw_vid_path = '/media/neuroai/T7/rPPG/rc_ContrastPhys/vv250'  # Path to raw video directory
    output_stmap_path = '/media/neuroai/T7/rPPG/STMap_raw/vv250'  # Path to output STmap directory

    create_stmap_for_videos(raw_vid_path, output_stmap_path)
