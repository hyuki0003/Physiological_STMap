"""
- Heart rate (HR) 를 추정 : calculate_HR
- Predicted PPG와 Ground truth PPG의 heart rate 오차 측정 : calculate_HR_metrics
 * MAE (Mean absolute error) : 평균 절대값 오차
 * RMSE (Root mean square error) : 평균 제곱근 오차
 * Pearson Correlation Coefficient : 피어슨 상관계수 -1 ~ 1 까지의 값을 가짐
                                     (HR의 선형성 반영 : 절댓값이 1에 가까울수록 선형적이고, 0에 가까울수록 비선형적)
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import welch, find_peaks
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_bvp_correlation(predictions, ground_truth):
    """
    Pearson correlation between predicted and GT BVP sequences.
    Args:
        predictions: np.ndarray of shape (B, T)
        ground_truth: np.ndarray of shape (B, T)
    Returns:
        avg_correlation: float
    """
    correlations = []

    for pred, gt in zip(predictions, ground_truth):
        if np.std(pred) < 1e-6 or np.std(gt) < 1e-6:
            continue  # skip flat signals
        corr = np.corrcoef(pred, gt)[0, 1]
        correlations.append(corr)

    if len(correlations) == 0:
        return np.nan

    return np.mean(correlations)


def safe_corrcoef(x, y):
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def diff_normalize_label(label):
    """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
    diff_label = np.diff(label, axis=0)
    # diff_label = np.append(diff_label, np.zeros(1), axis=0)
    # diff_label = np.insert(diff_label, 0, 0, axis=0)
    # diff_label = diff_label + label
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label

def standardized_label(label):
    """Z-score standardization for label signal."""
    label = label - np.mean(label)
    label = label / (np.std(label)+1e-8)
    label[np.isnan(label)] = 0
    return label
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def _next_power_of_2(x):
    """2의 가장 가까운 거듭제곱을 계산"""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _calculate_fft_mean_hr(ppg_signals, fs, low_pass=0.75, high_pass=2.5):
    hr_list = []
    for ppg_signal in ppg_signals:
        ppg_signal = np.asarray(ppg_signal)
        if ppg_signal.ndim == 1:
            ppg_signal = np.expand_dims(ppg_signal, 0)
        elif ppg_signal.ndim == 2 and ppg_signal.shape[0] > ppg_signal.shape[1]:
            ppg_signal = ppg_signal.T

        N = _next_power_of_2(ppg_signal.shape[1])
        f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
        hr_list.append(fft_hr)
    mean_hr = np.mean(hr_list)

    return np.array(hr_list)



def _calculate_peak_detection_hr(ppg_signals, fs):
    """Peak detection을 사용하여 PPG로부터 HR 계산"""

    hr_list = []
    for ppg_signal in ppg_signals:
        peaks, _ = find_peaks(ppg_signal, distance=fs / 2)  # fs/2의 간격으로 피크 탐지
        rr_intervals = np.diff(peaks) / fs
        hr = 60 / np.mean(rr_intervals)
        hr_list.append(hr)
    mean_hr = np.mean(hr_list)

    return hr_list

def calculate_metric_per_video(predictions, labels, fs=30):
    """Predicted PPG & ground truth PPG 의 HR 계산"""

    hr_pred = _calculate_fft_mean_hr(predictions, fs=fs)
    hr_label = _calculate_fft_mean_hr(labels, fs=fs)

    return hr_label, hr_pred


def _calculate_welch_hr(y, sr=30, min_hr=30, max_hr=180):
    num_segments, num_frames = y.shape
    hr_list = []

    for i in range(num_segments):
        freqs, power = welch(y[i], fs=sr, nfft=int(1e5 / sr), nperseg=np.min((len(y[i]) - 1, 256)))

        min_freq = min_hr / 60
        max_freq = max_hr / 60

        valid_freqs = freqs[(freqs > min_freq) & (freqs < max_freq)]
        valid_power = power[(freqs > min_freq) & (freqs < max_freq)]

        if len(valid_freqs) == 0 or len(valid_power) == 0:
            print(f"Warning: No valid frequencies found for segment {i}. Defaulting to mean HR.")
            hr_list.append(np.nan)  # 유효한 값이 없을 경우 NaN 추가
            continue

        peak_freq = valid_freqs[np.argmax(valid_power)]
        heart_rate = peak_freq * 60
        hr_list.append(heart_rate)

    hr_list = np.array(hr_list)

    if np.isnan(hr_list).all():  # 모든 값이 NaN인 경우
        raise ValueError("No valid heart rates calculated from the segments.")

    # NaN 값을 제외한 평균 계산 (필요에 따라)
    hr_list = np.nan_to_num(hr_list, nan=np.nanmean(hr_list))

    return hr_list

def calculate_hr_per_segment(y, sr=30, min_hr=30, max_hr=180):
    """
    각 세그먼트에서 개별적으로 심박수를 계산하는 함수.

    Args:
    y (np.ndarray): 입력 신호, (num_segments, num_frames) 형태의 배열.
    sr (int): 샘플링 속도 (Hz).
    min_hr (int): 최소 심박수 (bpm).
    max_hr (int): 최대 심박수 (bpm).

    Returns:
    List[float]: 각 세그먼트에서 계산된 심박수 목록.
    """
    num_segments, num_frames = y.shape
    hr_list = []

    for i in range(num_segments):
        # 각 세그먼트에 대해 심박수 계산
        freqs, power = welch(y[i], fs=sr, nfft=int(1e5 / sr), nperseg=np.min((len(y[i]) - 1, 256)))

        # BPM을 Hz로 변환
        min_freq = min_hr / 60
        max_freq = max_hr / 60

        # 원하는 심박수 범위 내의 주파수와 파워 찾기
        valid_freqs = freqs[(freqs > min_freq) & (freqs < max_freq)]
        valid_power = power[(freqs > min_freq) & (freqs < max_freq)]

        # 유효한 주파수와 파워가 있는지 확인
        if len(valid_freqs) == 0 or len(valid_power) == 0:
            print(f"Warning: No valid frequencies found for segment {i}")
            hr_list.append(np.nan)  # 유효한 주파수가 없으면 NaN을 추가
            continue

        # 유효한 범위 내에서 최대 파워를 가지는 주파수 찾기
        peak_freq = valid_freqs[np.argmax(valid_power)]

        # 주파수를 BPM으로 변환
        heart_rate = peak_freq * 60
        hr_list.append(heart_rate)

    return np.mean(hr_list)


def calculate_pearson_correlation_bvp(predictions, ground_truth, fs=30):
    """예측된 심박수와 실제 심박수 간의 Pearson 상관계수 계산"""
    mean_signal1 = np.mean(predictions, axis=0, keepdims=True)
    mean_signal2 = np.mean(ground_truth, axis=0, keepdims=True)

    mean_signal1 = np.ravel(mean_signal1)
    mean_signal2 = np.ravel(mean_signal2)

    correlation, _ = pearsonr(mean_signal1, mean_signal2)

    return correlation

def calculate_hr(ppg_signals, fs=30, low_pass=0.75, high_pass=2.5):
    hr_list = []
    for ppg_signal in ppg_signals:
        ppg_signal = np.asarray(ppg_signal)

        if ppg_signal.ndim == 0 or ppg_signal.size == 0:
            continue

        if ppg_signal.ndim == 1:
            if ppg_signal.shape[0] == 0:
                continue
            ppg_signal = np.expand_dims(ppg_signal, 0)

        elif ppg_signal.ndim == 2:
            if ppg_signal.shape[0] > ppg_signal.shape[1]:
                ppg_signal = ppg_signal.T
            if ppg_signal.shape[1] == 0:
                continue

        else:
            continue

        try:
            N = _next_power_of_2(ppg_signal.shape[1])
            f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
            fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
            mask_ppg = np.take(f_ppg, fmask_ppg)
            mask_pxx = np.take(pxx_ppg, fmask_ppg)
            fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
            hr_list.append(fft_hr)
        except Exception as e:
            print(f"[calculate_hr] Skip invalid PPG segment due to: {e}")
            continue

    if len(hr_list) == 0:
        return np.nan

    return np.mean(hr_list)

def calculate_hr_metrics(predictions, ground_truth, fs=30):
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    if predictions.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("Predictions and ground_truth must both be 2D arrays (B, T)")

    pred_hrs = np.array([calculate_hr([pred], fs) for pred in predictions])
    gt_hrs = np.array([calculate_hr([gt], fs) for gt in ground_truth])

    valid_idx = ~np.isnan(pred_hrs) & ~np.isnan(gt_hrs)
    pred_hrs = pred_hrs[valid_idx]
    gt_hrs = gt_hrs[valid_idx]

    if len(pred_hrs) == 0 or len(gt_hrs) == 0:
        print("[calculate_hr_metrics] Warning: No valid HR pairs. Skipping metric calculation.")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mae = mean_absolute_error(gt_hrs, pred_hrs)
    mse = mean_squared_error(gt_hrs, pred_hrs)
    rmse = np.sqrt(mse)
    correlation = safe_corrcoef(gt_hrs, pred_hrs)  # ✅ 여기가 핵심
    snr = calculate_snr(pred_hrs, gt_hrs)

    return mae, mse, rmse, correlation, snr





def calculate_snr(predictions, ground_truth):
    """Calculate Signal-to-Noise Ratio (SNR)"""
    pred_hrs = np.array(predictions)
    gt_hrs = np.array(ground_truth)

    # Calculate the power of the signal and the noise
    signal_power = np.mean(gt_hrs ** 2)
    noise_power = np.mean((gt_hrs - pred_hrs) ** 2)

    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)

    return snr
def calculate_std(metrics):

    return np.std(metrics, axis=0)
