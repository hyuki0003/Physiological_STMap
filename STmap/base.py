from scipy.interpolate import splrep, splev
import face_alignment
import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from scipy.signal import butter, filtfilt

def make_route(path: str):
    try:
        os.makedirs(path, exist_ok=True)  # Create the directory and ignore if it already exists
        print(f"Directory created: {path}")
    except Exception as e:
        print(f"Error creating directory {path}: {e}")

class BasePreprocess:
    def __init__(self, raw_data_path:str, preprocess_data_path:str, img_size:int=128, m:int=16, fs:float=30., fl:float=0.4, fh:float=2.5, order:int=8, device:str='cuda:2', do_stmap:bool=True, stmap_type:int=2):
        """Preprocess from facial videos and bvps.

        Args:
            Input:
                raw_data_path (str): Raw data path.
                preprocess_data_path (str): Preprocessed data path.
                img_size (int) : image size.
                m (int): num of pixels to patchify over width and height from a frame.
                fs (float): sampling frequency (or frame per second).
                fl (float) : low cutoff frequency for butterworth band pass filter.
                fh (float): high cutoff frequency for butterworth band pass filter.
                order (int): order of butterworth band pass filter.
                device (str): device to use.
                do_stmap (bool): whether to use stmap.
                stmap_type (int): type of stmap. [0: rgb stmap, 1: yuv stmap, 2: both]

            Member Variables:
                self.x (list): file paths for spatial-temporal (ST) map if do_stmap (bool) == True else face-detected videos
                self.y (list): file paths for ground-truth (GT) blood volume pulse (BVP)
                self.abnormal_files : Unreadable (abnormal) data file names
        """

        video_metadata = self._get_video_metadata(
            raw_data_path)  # index, path, and subject information of video data

        self.abnormal_files = [] # contains abnormal video files' names

        # Identifying Landmarks and Aligning Face
        align_video_path = os.path.join(preprocess_data_path, 'align')
        self._align(video_metadata, align_video_path, fs, device, img_size)
        print(f"\nAmong {len(video_metadata)} facial videos, {len(video_metadata)-len(self.abnormal_files)} videos are aligned and restored in .{align_video_path}.")
        print(f"But, {len(self.abnormal_files)} abnormal video files are detected. \n Files: {self.abnormal_files}")
        video_metadata = self._get_video_metadata(align_video_path)

        # Synchronize the bvps to videos
        bvps_path = os.path.join(preprocess_data_path, 'vts_bvps')
        if not os.path.exists(bvps_path) or not os.listdir(bvps_path):
            self._sync(preprocess_data_path, raw_data_path)

        # Make stmap
        stmap_path = os.path.join(preprocess_data_path, 'stmap')
        if not os.path.exists(stmap_path) or not os.listdir(stmap_path):
            if do_stmap:
                self._stmap(video_metadata, stmap_path, m, fs, fl, fh, order)

        input_data_path = None
        if do_stmap:
            if stmap_type == 0:
                input_data_path = os.path.join(stmap_path, 'rgb')
            elif stmap_type == 1:
                input_data_path = os.path.join(stmap_path, 'yuv')
            elif stmap_type == 2:
                input_data_path = os.path.join(stmap_path, 'both')
            else:
                raise NotImplementedError(f"stmap_type {stmap_type} should be a value in [0: rgb, 1: yuv, 2: both].")
        else:
            input_data_path = align_video_path

        self.x = sorted(
            [os.path.join(input_data_path, f) for f in os.listdir(input_data_path) if f.endswith(('.npy'))])

        ground_truth_path = os.path.join(preprocess_data_path, 'vts_bvps')
        self.y = sorted(
            [os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path) if f.endswith('.npy')])

        return


    def _get_video_metadata(self, data_path):
        """Returns raw data directories under the path.

        Args:
            data_path(str): a list of video_files.
        """
        return


    def _sync(self, preprocess_data_path, raw_data_path):
        """
        Synchronizes the BVPs to Facial video

        """
        return

    def _align(self, video_metadata, align_video_path, fs, device, img_size, total_failure_rate=0.05, consecutive_failure_count=9):

        make_route(align_video_path)
        model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        for infos in tqdm(video_metadata):
            raw_data_name = infos["index"]
            raw_data_path = infos["path"]
            save_filename = raw_data_name + '.mp4'
            save_filepath = os.path.join(align_video_path, save_filename)
            if os.path.exists(save_filepath):
                continue

            # Landmark detection and processing abnormal landmark
            VidObj = cv2.VideoCapture(raw_data_path)
            VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)

            frames = []
            abnormal_indices = []
            lmks = []

            cnt=0
            missing_count =0
            consecutive_missing_count = 0
            error = False
            while True:
                success, frame = VidObj.read()
                if not success: break

                # GPU에 바로 올려서 처리
                frame_tensor = torch.tensor(frame, dtype=torch.uint8, device=device)
                frames.append(frame_tensor)

                with torch.no_grad():
                    lmk = model.get_landmarks(frame_tensor)

                if lmk is None:
                    abnormal_indices.append(cnt)
                    missing_count+=1
                    consecutive_missing_count+=1
                else:
                    lmks.append(lmk[0].reshape(136))
                    consecutive_missing_count=0

                # Landmark error
                if consecutive_missing_count >= consecutive_failure_count:
                    print(f"Discarding {raw_data_name} - Too many consecutive failures during detecting landmarks, {consecutive_missing_count} frames")
                    VidObj.release()
                    error =True
                    break

                cnt += 1
            VidObj.release()

            n_frames = len(frames)

            # Frame error
            if n_frames == 0:
                self.abnormal_files.append(save_filename)
                print(f"Discarding {raw_data_name} - Cannot read frames")
                continue

            # Landmark error
            if missing_count >= n_frames*total_failure_rate:
                print(f"Discarding {raw_data_name} - Too many missing landmarks {missing_count} out of {n_frames} frames")
                error = True

            if error: continue

            frames = torch.stack(frames).cpu().numpy()


            frame_indices = np.linspace(0, n_frames, n_frames, dtype=int)
            normal_indices = np.delete(frame_indices, abnormal_indices)
            lmks = np.asarray(lmks)

            interpolated_lmks = []
            print(f"*** {raw_data_name} video's landmarks are interpolated  ***\nNum of Landmark: {len(lmks[0])} Raw data name: {raw_data_name}\n abnormal_indices: {abnormal_indices}")

            for i in range(136):
                ith_lmks = lmks[:, i]
                spline_representation = splrep(normal_indices, ith_lmks)
                interpolated_lmk = splev(frame_indices, spline_representation)
                interpolated_lmk= self._smooth_with_edge_padding(interpolated_lmk, 10).astype(int)
                interpolated_lmks.append(interpolated_lmk)

            lmks = np.array(interpolated_lmks).T

            # Face Alignment
            aligned_faces = list()

            ## Precompute transformation for face alignment
            # dst_points = np.array([[0, 48], [128, 48], [64, 128]], dtype=np.float32)
            dst_points = np.array([[20,48], [108,48], [64,128]], dtype=np.float32)
            for i, frame in enumerate(frames):
                lmk = lmks[i].reshape(-1, 2)
                ## Face alignment using predefined points
                # src_points = np.array([lmk[1], lmk[15], lmk[8]], dtype=np.float32)
                src_points = np.array([lmk[36], lmk[45], lmk[8]], dtype=np.float32)
                M = cv2.getAffineTransform(src_points, dst_points)
                face_aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
                aligned_faces.append(face_aligned[:img_size, :img_size])

            aligned_faces = np.asarray(aligned_faces)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 저장을 위한 코덱 설정
            out = cv2.VideoWriter(save_filepath, fourcc, fs, (img_size, img_size))

            for frame in aligned_faces:
                out.write(frame)

            out.release()
            cv2.destroyAllWindows()

        return


    def _stmap(self, video_metadata, stmap_path, m, fs, fl, fh, order):
        rgb_stmap_path = os.path.join(stmap_path, "rgb")
        make_route(rgb_stmap_path)

        yuv_stmap_path = os.path.join(stmap_path, "yuv")
        make_route(yuv_stmap_path)

        both_stmap_path = os.path.join(stmap_path, "both")
        make_route(both_stmap_path)

        for infos in tqdm(video_metadata):
            # print(f"Set Landmarks of the file {infos['index']} at time step {idx+1}")
            data_name = infos["index"]
            data_path = infos["path"]

            cap = cv2.VideoCapture(data_path)

            frames = []

            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            frames = np.array(frames, dtype=np.float32)
            T, H, W, C = frames.shape

            # b, a = butter(order, [fl / (fs / 2), fh / (fs / 2)], 'bandpass')
            # frames = filtfilt(b, a, frames.reshape(T, -1), axis=0).reshape(T, H, W, C)
            rgbs = (frames - frames.mean(axis=(1, 2), keepdims=True) + 1e-12) / (
                    frames.std(axis=(1, 2), keepdims=True) + 1e-12)
            rgb_stmap = self._compute_stmap(rgbs, T, H, W, C, m, rgb_stmap_path, data_name) # for rgb stmap [C, H, W, T]

            yuvs = np.array([self._RGB2YUV(frame) for frame in frames])  # Apply transformation to all frames
            yuvs = (yuvs - yuvs.mean(axis=(1, 2), keepdims=True) + 1e-12) / (
                    yuvs.std(axis=(1, 2), keepdims=True) + 1e-12)
            yuv_stmap = self._compute_stmap(yuvs, T, H, W, C, m, yuv_stmap_path, data_name) # for yuv stmap [C, H, W, T]

            both_stmap = np.concatenate((rgb_stmap, yuv_stmap), axis=0)
            write_path = os.path.join(both_stmap_path, f"{data_name}.npy")
            np.save(write_path, both_stmap)

        return

    def _compute_stmap(self, frames,T, H, W, C, m, save_path, data_name):
        Hm, Wm = H // m, W // m

        # Reshape to divide image into (m, m) regions
        frames_reshaped = frames.reshape(T, Hm, m, Wm, m, C)

        # Compute mean over (m, m) blocks
        stmap = frames_reshaped.mean(axis=(2, 4))  # Averaging over (m, m) regions
        # Normalization over each frame to mitigate influence of momentary baseline.
        stmap = stmap.transpose((3, 1, 2, 0))  # T, H, W, C -> C, H, W, T

        write_path = os.path.join(save_path, f"{data_name}.npy")
        np.save(write_path, stmap)

        return stmap


    def _RGB2YUV(self, RGBimg):
        """BT.709 Standard"""
        transformation_matrix = np.array([
            [0.2126, 0.7152, 0.0722],  # Y
            [-0.09991, -0.33609, 0.436],  # U
            [0.615, -0.55861, -0.05639]  # V
        ])

        img_rgb_reshaped = RGBimg.reshape(-1, 3).astype(np.float32)
        img_yuv_reshaped = img_rgb_reshaped @ transformation_matrix.T
        img_yuv_reshaped[:, 1:] += 128  # U, V 채널 범위 조정

        return np.clip(img_yuv_reshaped.reshape(RGBimg.shape), 0, 255).astype(np.uint8)


    def _YUV2RGB(self, YUVimg):
        """BT.709 Standard"""
        inverse_matrix = np.array([
            [1.0, 0.0, 1.28033],  # R
            [1.0, -0.21482, -0.38059],  # G
            [1.0, 2.12798, 0.0]  # B
        ])

        img_yuv_reshaped = YUVimg.reshape(-1, 3).astype(np.float32)
        img_yuv_reshaped[:, 1:] -= 128  # U, V 채널 복구
        img_rgb_reshaped = img_yuv_reshaped @ inverse_matrix.T

        return np.clip(img_rgb_reshaped.reshape(YUVimg.shape), 0, 255).astype(np.uint8)


    def _smooth_with_edge_padding(SELF,data, window_size=5):
        pad_size = window_size // 2  # 앞뒤로 추가할 패딩 크기
        padded_data = np.pad(data, (pad_size, pad_size), mode='edge')  # 가장자리 값 반복 패딩
        smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
        return smoothed_data
