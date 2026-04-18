import os
import glob
from tqdm import tqdm
import face_alignment
import torch
import cv2  # OpenCV를 임포트하여 BGR을 RGB로 변환
import stmap_generator as su  # STmap 유틸리티 함수들을 임포트


def initialize_cuda():
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA is available. Number of devices: {device_count}")
            return 'cuda'
        else:
            print("CUDA is not available.")
            return 'cpu'
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        return 'cpu'


def create_video_from_images(images_dir, video_path, fps=30):
    """이미지 파일들을 하나의 동영상으로 변환합니다."""
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    if not image_files:
        print(f"이미지 파일을 찾을 수 없습니다: {images_dir}")
        return

    # 첫 번째 이미지 파일을 읽어 비디오의 크기 설정
    frame = cv2.imread(image_files[0])
    if frame is None:
        print(f"이미지 파일을 읽을 수 없습니다: {image_files[0]}")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image_path in tqdm(image_files, desc="Creating Video"):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"이미지 파일을 읽을 수 없습니다: {image_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"비디오 파일을 생성했습니다: {video_path}")


def process_video_for_stmap(raw_video_path, subdir, output_stmap_root_path, device):
    try:
        input_filename = os.path.splitext(os.path.basename(raw_video_path))[0]
        output_dir = os.path.join(output_stmap_root_path, subdir)
        os.makedirs(output_dir, exist_ok=True)

        raw_frames = su.get_frames(raw_video_path)
        if len(raw_frames) == 0:
            print(f"비디오에서 프레임을 찾을 수 없습니다: {raw_video_path}")
            return

        # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        # lmks = su.get_landmarks(fa, raw_frames)
        # if len(lmks) == 0:
        #     print(f"비디오에서 랜드마크를 찾을 수 없습니다: {raw_video_path}")
        #     return
        #
        # aligned_faces = su.align_face(raw_frames, lmks)

        stmap_yuv = su.STmap(raw_frames)
        stmap_rgb = su.YUV2RGB(stmap_yuv)
        su.save_STmap(stmap_yuv, output_dir, f'vid_stmap_yuv.png')
        su.save_STmap(stmap_rgb, output_dir, f'vid_stmap_rgb.png', convert_to_bgr=False)
        print(f"{subdir}의 {input_filename}에 대한 ST맵을 처리하고 저장했습니다.")

    except Exception as e:
        print(f"비디오 {raw_video_path} 처리 실패: {e}")


def create_stmap_for_videos(raw_img_dir_path, output_stmap_root_path, device):
    for root, dirs, files in os.walk(raw_img_dir_path):
        for dir_name in dirs:
            images_dir = os.path.join(root, dir_name)
            video_path = os.path.join(images_dir, f"{dir_name}_processed.avi")

            # create_video_from_images(images_dir, video_path)
            process_video_for_stmap(video_path, dir_name, output_stmap_root_path, device)


if __name__ == "__main__":
    device = initialize_cuda()
    raw_img_path = '/media/neuroai/T7/rPPG/rc_ContrastPhys/PURE'  # 원시 이미지 파일 디렉토리 경로
    output_stmap_path = '/media/neuroai/T7/rPPG/STMap_raw/PURE'  # 출력 STMap 디렉토리 경로

    create_stmap_for_videos(raw_img_path, output_stmap_path, device)
