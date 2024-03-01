import numpy as np
from scipy.signal import find_peaks
import csv

def _compute_mean_features(window):
    mean = np.mean(window, axis=0)
    return mean

def _median_feature(window):
    median = np.median(window)
    return median

def _fft_feature(window):
    fft_win = np.fft.rfft(window)
    refined_fft_win = np.abs(fft_win)  # Get the absolute values of the FFT coefficients
    max_coeff = np.max(refined_fft_win)  # Get the maximum FFT coefficient
    return max_coeff

def _entropy_feature(window):
    hist_win = np.histogram(window, bins=10)
    distribution = hist_win[0] / window.size  # Normalize the distribution
    entropy_val = -np.sum(distribution * np.log2(distribution + np.finfo(float).eps))  # Calculate entropy
    return entropy_val

def _peak_feature(window):
    peaks = []
    mean_signal = np.average(window)
    peak_arr, _ = find_peaks(window)
    for p in range (len(peak_arr)):
        if peak_arr[p] >= mean_signal:
            peaks.append(peak_arr[p])
    return len(peaks)



def extract_features(window):
    x = []
    feature_names = []
    win = np.array(window)
    
    x_arr = win[:,0]
    y_arr = win[:,1]
    z_arr = win[:,2]
    mag_window = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)  
    
    x.append(_compute_mean_features(win[:,0]))
    feature_names.append("x_mean")

    x.append(_compute_mean_features(win[:,1]))
    feature_names.append("y_mean")

    x.append(_compute_mean_features(win[:,2]))
    feature_names.append("z_mean")
    
    x.append(_median_feature(mag_window))
    feature_names.append("magnitude_median")
    
    x.append(_fft_feature(win[:,0]))
    feature_names.append("x_fft")
    
    x.append(_fft_feature(win[:,1]))
    feature_names.append("y_fft")
    
    x.append(_fft_feature(win[:,2]))
    feature_names.append("z_fft")
    
    x.append(_fft_feature(mag_window))
    feature_names.append("magnitude_fft")
    
    x.append(_entropy_feature(mag_window))
    feature_names.append("magnitude_entropy")
    
    x.append(_peak_feature(mag_window))
    feature_names.append("magnitude_number of peaks")
    
    feature_vector = list(x)
    return feature_names, feature_vector
    