import os
import glob
from scipy.interpolate import PchipInterpolator
from .base import BasePreprocess
from .base import make_route
import json
import numpy as np


class vvPreprocess(BasePreprocess):
    def __init__(self, raw_data_path:str, preprocess_data_path:str, img_size:int=128, m:int=16, fs:float=30., fl:float=0.4, fh:float=2.5, order:int=8, device:str='cuda:2', stmap:bool=True, stmap_type:int=2):
        super().__init__(raw_data_path,preprocess_data_path, img_size, m, fs, fl, fh, order, device, stmap, stmap_type)

    def _get_video_metadata(self, data_path):
        data_dirs = glob.glob(os.path.join(data_path, "*.mp4"))
        if not data_dirs:
            raise ValueError(data_path + " data paths empty!")
        video_metadata = list()
        for data_dir in data_dirs:
            root, _ = os.path.splitext(data_dir)  # without extension
            index = os.path.split(root)[-1]  # filename
            # subject = index[0:-2]  # filename without number
            video_metadata.append({"index": index, "path": data_dir})
        return video_metadata

    def _get_bp(self,  preprocess_data_path, raw_data_path):
        bp_save_path = preprocess_data_path + "/bp/"
        make_route(bp_save_path)

        for dirpath, dirnames, filenames in os.walk(raw_data_path):
            for filename in filenames:
                if filename.endswith('.json'):
                    if filename == "READ_ME.json":
                        continue
                    with open(os.path.join(dirpath, filename)) as json_file:
                        data = json.load(json_file)
                        for scenario in data['scenarios']:
                            # Check if the recording file exists in the directory (some videos were unreadable so they had to be deleted)
                            recording_link = scenario['recordings']['RGB']['filename']
                            if recording_link not in self.abnormal_files:
                                save_filename = str(recording_link)[:-4]
                                print(save_filename)

                                sbp = scenario['recordings']['bp_sys']['value'] if 'bp_sys' in scenario[
                                    'recordings'] else -1
                                dbp = scenario['recordings']['bp_dia']['value'] if 'bp_dia' in scenario[
                                    'recordings'] else -1
                                if sbp == -1 or dbp == -1:
                                    continue
                                bp_values_array = np.array([sbp, dbp])
                                np.save(bp_save_path + save_filename + ".npy", bp_values_array)
        return


    def _sync(self, preprocess_data_path, raw_data_path):
        ts_bvps_save_path = preprocess_data_path + "/ts_bvps/"
        vts_bvps_save_path = preprocess_data_path + "/vts_bvps/"

        make_route(ts_bvps_save_path)
        make_route(vts_bvps_save_path)

        for dirpath, dirnames, filenames in os.walk(raw_data_path):
            for filename in filenames:
                if filename.endswith('.json'):
                    if filename == "READ_ME.json":
                        continue
                    with open(os.path.join(dirpath, filename)) as json_file:
                        data = json.load(json_file)
                        for scenario in data['scenarios']:
                            # Check if the recording file exists in the directory (some videos were unreadable so they had to be deleted)
                            recording_link = scenario['recordings']['RGB']['filename']
                            if recording_link not in self.abnormal_files:
                                ts_list = []
                                ppg_values_list = []
                                vts_list = []
                                save_filename = str(recording_link)[:-4]

                                for item in scenario['recordings']['RGB']['timeseries']:
                                    if item[0] not in vts_list and item[1] <= 900:
                                        vts_list.append(item[0])

                                vts_values_array = np.array(vts_list)

                                for item in scenario['recordings']['ppg']['timeseries']:
                                    if item[0] not in ts_list:
                                        ts_list.append(item[0])  # timestamp by milliseconds
                                        ppg_values_list.append(item[1])  # ppg sampling point

                                ts_values_array = np.asarray(ts_list)
                                ppg_array = np.asarray(ppg_values_list)
                                # ppg_array = np.vstack((ts_values_array, ppg_values_array))

                                # Cubic Hermite interpolation to synchronize PPGs into Frames
                                pchip_interpolator = PchipInterpolator(ts_values_array, ppg_array)
                                resampled_ppg = pchip_interpolator(vts_values_array)
                                min_val = resampled_ppg.min()
                                resampled_ppg = (resampled_ppg - min_val) / (resampled_ppg.max()-min_val) # min-max norm

                                np.save(vts_bvps_save_path + save_filename+ ".npy", resampled_ppg)
                                np.save(ts_bvps_save_path + save_filename + ".npy", ppg_array)
