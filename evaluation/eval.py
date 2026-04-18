import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from evaluation.eval_dataset_loader import SingleSubjectData
from evaluation.temp import SingleSubjectDataFull
from utils.metrics import metrics

from utils.filter.kalman import apply_kalman_filter
from utils.filter.low_pass import low_pass_filter

from scipy.signal import welch

def plot_psd(signal, fs, label):
    f, Pxx = welch(signal, fs, nperseg=fs*2)
    plt.semilogy(f, Pxx, label=label)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.grid()
    plt.legend()

def compare_psd(predictions, ground_truth, fs):
    plt.figure(figsize=(10, 6))

    # 예측된 신호의 PSD 계산 및 시각화
    plot_psd(predictions.flatten(), fs, label='Predicted BVP')

    # 실제 신호의 PSD 계산 및 시각화
    plot_psd(ground_truth.flatten(), fs, label='Ground Truth BVP')

    plt.title('Power Spectral Density (PSD) of Predicted and Ground Truth BVP')
    plt.show()

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for feature_map, bvp in dataloader:
            feature_map = feature_map.to(device)
            outputs = model(feature_map)
            predictions.append(outputs.cpu().numpy())
            ground_truth.append(bvp.numpy())

    if predictions and ground_truth:
        predictions = np.concatenate(predictions, axis=0)
        ground_truths = np.concatenate(ground_truth, axis=0)
    else:
        predictions = np.array([])
        ground_truths = np.array([])

    return predictions, ground_truths

def calculate_hr(signal):
    """HR 계산 함수"""
    # Signal의 shape 확인 및 변환
    if signal.ndim == 1:
        # 1차원 배열인 경우, (1, length) 형식의 2차원 배열로 변환
        signal = np.expand_dims(signal, axis=0)
    elif signal.ndim != 2:
        raise ValueError(f"Unexpected signal shape: {signal.shape}. Expected 2D array.")


    return metrics._calculate_welch_hr(signal)

def zero_mean(signal):
    """신호를 제로민으로 변환"""
    mean = np.mean(signal, axis=-1, keepdims=True)
    zero_mean_signal = signal - mean
    return zero_mean_signal

def normalize_signals(predictions, ground_truth):
    predictions_zero_mean = zero_mean(predictions)
    ground_truth_zero_mean = zero_mean(ground_truth)

    predictions_normalized = []
    ground_truth_normalized = []

    for pred, gt in zip(predictions_zero_mean, ground_truth_zero_mean):
        pred_mean, pred_std = np.mean(pred), np.std(pred)
        gt_mean, gt_std = np.mean(gt), np.std(gt)

        if pred_std == 0:
            pred_normalized = pred - pred_mean
        else:
            pred_normalized = (pred - pred_mean) / pred_std

        if gt_std == 0:
            gt_normalized = gt - gt_mean
        else:
            gt_normalized = (gt - gt_mean) / gt_std

        predictions_normalized.append(pred_normalized)
        ground_truth_normalized.append(gt_normalized)

    predictions_normalized = np.array(predictions_normalized)
    ground_truth_normalized = np.array(ground_truth_normalized)

    return predictions_normalized, ground_truth_normalized

def calculate_bvp_metrics(predictions, ground_truth):
    """BVP 신호에 대한 MAE, RMSE, Pearson 계산"""
    predictions_flat = predictions.flatten()
    ground_truth_flat = ground_truth.flatten()

    mae = mean_absolute_error(ground_truth_flat, predictions_flat)
    mae_std = np.std(np.abs(ground_truth_flat - predictions_flat))

    rmse = np.sqrt(mean_squared_error(ground_truth_flat, predictions_flat))
    rmse_std = np.std((ground_truth_flat - predictions_flat) ** 2)

    pearson_corr, _ = pearsonr(predictions_flat, ground_truth_flat)

    return mae, mae_std, rmse, rmse_std, pearson_corr

def calculate_hr_metrics(hr_predictions, hr_ground_truth):
    """HR 신호에 대한 MAE 및 RMSE 계산"""
    hr_predictions_flat = hr_predictions.flatten()
    hr_ground_truth_flat = hr_ground_truth.flatten()

    mae = mean_absolute_error(hr_ground_truth_flat, hr_predictions_flat)
    mae_std = np.std(np.abs(hr_ground_truth_flat - hr_predictions_flat))

    rmse = np.sqrt(mean_squared_error(hr_ground_truth_flat, hr_predictions_flat))
    rmse_std = np.std((hr_ground_truth_flat - hr_predictions_flat) ** 2)

    # Pearson 상관계수는 계산하지 않음
    return mae, mae_std, rmse, rmse_std

def bland_altman_plot(predictions, ground_truth, save_path=None):
    import matplotlib.pyplot as plt

    mean = np.mean([predictions, ground_truth], axis=0)
    diff = predictions - ground_truth
    md = np.mean(diff)
    sd = np.std(diff)

    plt.figure(figsize=(10, 6))
    plt.scatter(mean, diff, s=20, alpha=0.5)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='red', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='red', linestyle='--')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean of predictions and ground truth')
    plt.ylabel('Difference between predictions and ground truth')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def bvp_evaluation(predictions, ground_truth, save_path=None):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.numpy()

    mae, mae_std, rmse, rmse_std, pearson_corr = calculate_bvp_metrics(predictions, ground_truth)

    print(f"BVP MAE: {mae} ± {mae_std}")
    print(f"BVP RMSE: {rmse} ± {rmse_std}")
    print(f"BVP Pearson Correlation: {pearson_corr}")

    # bland_altman_plot(predictions, ground_truth, save_path)

    return mae, mae_std, rmse, rmse_std, pearson_corr

def hr_evaluation(predictions, ground_truth, save_path=None):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.numpy()

    # 전체 시퀀스에 대해 HR을 계산
    hr_predictions = calculate_hr(predictions)
    hr_ground_truth = calculate_hr(ground_truth)
    print("p", np.mean(hr_predictions))
    print("g", np.mean(hr_ground_truth))

    mae2= np.abs(np.mean(hr_predictions)-np.mean(hr_ground_truth))

    # HR에 대한 MAE 및 RMSE 계산
    mae, mae_std, rmse, rmse_std = calculate_hr_metrics(hr_predictions, hr_ground_truth)

    print(f"HR MAE: {mae2}")
    print(f"HR RMSE: {rmse}")

    # bland_altman_plot(hr_predictions, hr_ground_truth, save_path)

    return mae2, mae_std, rmse, rmse_std


def evaluate_all_subjects(subject_dirs, save_dir, model, device, args):
    maes_bvp, rmse_list_bvp, pearson_list_bvp = [], [], []
    maes_hr, rmse_list_hr, pearson_list_hr = [], [], []

    for i, subject_dir in enumerate(subject_dirs):
        print(f"Processing subject {i + 1}/{len(subject_dirs)}: {os.path.basename(subject_dir)}")

        dataset = SingleSubjectData(
            subject_dir=subject_dir,
            version=args.version,
            channels=args.channels,
            STMap1=args.STMap1,
            STMap2=args.STMap2,
            frames_num=args.frame_num,
            height=args.height,
            dataName=args.dataName,

        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        predictions, ground_truth = evaluate(model, dataloader, device)

        print(np.shape(predictions))
        print(np.shape(ground_truth))

        predictions = predictions.mean(axis=1, keepdims=True).squeeze(1)

        predictions, ground_truth = normalize_signals(predictions, ground_truth)

        predictions = torch.tensor(predictions)
        ground_truth = torch.tensor(ground_truth)

        predictions = metrics.butter_bandpass_filter(predictions, lowcut=0.75, highcut=2.5, fs=30, order=5)
        ground_truth = metrics.butter_bandpass_filter(ground_truth, lowcut=0.75, highcut=2.5, fs=30, order=5)

        # BVP 평가
        save_path_bvp = os.path.join(save_dir, f'{os.path.basename(subject_dir)}_bvp_bland_altman.png')
        mae_bvp, mae_std_bvp, rmse_bvp, rmse_std_bvp, pearson_corr_bvp = bvp_evaluation(predictions, ground_truth,
                                                                                        save_path_bvp)

        maes_bvp.append(mae_bvp)
        rmse_list_bvp.append(rmse_bvp)
        pearson_list_bvp.append(pearson_corr_bvp)

        # HR 평가
        save_path_hr = os.path.join(save_dir, f'{os.path.basename(subject_dir)}_hr_bland_altman.png')
        mae_hr, mae_std_hr, rmse_hr, rmse_std_hr = hr_evaluation(predictions, ground_truth,
                                                                                  save_path_hr)

        maes_hr.append(mae_hr)
        rmse_list_hr.append(rmse_hr)

        compare_psd(predictions, ground_truth, fs=30)

    # 전체 서브젝트에 대한 BVP 평가 결과
    overall_mae_bvp = np.mean(maes_bvp)
    overall_rmse_bvp = np.mean(rmse_list_bvp)
    overall_pearson_bvp = np.mean(pearson_list_bvp)


    print(f"Overall BVP MAE: {overall_mae_bvp}")
    print(f"Overall BVP RMSE: {overall_rmse_bvp}")
    print(f"Overall BVP Pearson Correlation: {overall_pearson_bvp}")

    # 전체 서브젝트에 대한 HR 평가 결과
    overall_mae_hr = np.mean(maes_hr)
    overall_rmse_hr = np.mean(rmse_list_hr)

    print(f"Overall HR MAE: {overall_mae_hr}")
    print(f"Overall HR RMSE: {overall_rmse_hr}")

def plot_comparison(predictions, ground_truth):
    plt.figure(figsize=(14, 7))

    predictions_flat = predictions.flatten()
    ground_truth_flat = ground_truth.flatten()

    plt.subplot(3, 1, 1)
    plt.plot(predictions_flat, label='Predicted BVP', color='blue')
    plt.title('Predicted BVP')
    plt.xlabel('Time')
    plt.ylabel('BVP Value')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(ground_truth_flat, label='Ground Truth BVP', color='red')
    plt.title('Ground Truth BVP')
    plt.xlabel('Time')
    plt.ylabel('BVP Value')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(predictions_flat, label='Predicted BVP', color='blue')
    plt.plot(ground_truth_flat, label='Ground Truth BVP', color='red')
    plt.title('Comparison of Predicted and Ground Truth BVP')
    plt.xlabel('Time')
    plt.ylabel('BVP Value')
    plt.legend()

    plt.tight_layout()
    plt.show()