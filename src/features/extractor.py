import numpy as np
import librosa

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, n_fft=frame_length, hop_length=hop_length)
    # We want (20, frames) -> (frames, 20) then flattened or just flattened
    return np.ravel(mfcc.T)

def extract_features(data, sr=22050):
    # This matches the 2,376 feature extraction from the 97.25% accurate notebook
    # 22 features per frame (1 ZCR, 1 RMSE, 20 MFCC)
    # Approx 108 frames for 2.5s audio (22050 * 2.5 / 512)
    result = np.array([])
    
    # 1. ZCR
    result = np.hstack((result, zcr(data)))
    
    # 2. RMSE
    result = np.hstack((result, rmse(data)))
    
    # 3. MFCC
    result = np.hstack((result, mfcc(data, sr)))
    
    return result

def get_features(path):
    # duration and offset as per notebook
    # 2.5s at 22050 sr is 55125 samples
    target_length = 55125 
    
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # Ensure fixed length
    if len(data) < target_length:
        data = librosa.util.pad_center(data, size=target_length)
    else:
        data = data[:target_length]
    
    # 1. Original
    res1 = extract_features(data, sample_rate)
    # Check if we got exactly 2376 features (22 * 108 frames)
    # 55125 / 512 is 107.66... result frames should be 108.
    if len(res1) != 2376:
        # Emergency fall-back padding if frames differ slightly
        res1 = np.pad(res1, (0, 2376 - len(res1)), 'constant') if len(res1) < 2376 else res1[:2376]

    result = np.array(res1)
    
    # 2. Noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    if len(res2) != 2376:
        res2 = np.pad(res2, (0, 2376 - len(res2)), 'constant') if len(res2) < 2376 else res2[:2376]
    result = np.vstack((result, res2))
    
    # 3. Pitch
    pitched_audio = pitch(data, sample_rate)
    res3 = extract_features(pitched_audio, sample_rate)
    if len(res3) != 2376:
        res3 = np.pad(res3, (0, 2376 - len(res3)), 'constant') if len(res3) < 2376 else res3[:2376]
    result = np.vstack((result, res3))
    
    # 4. Pitch + Noise
    pitched_noise_audio = noise(pitched_audio)
    res4 = extract_features(pitched_noise_audio, sample_rate)
    if len(res4) != 2376:
        res4 = np.pad(res4, (0, 2376 - len(res4)), 'constant') if len(res4) < 2376 else res4[:2376]
    result = np.vstack((result, res4))
    
    return result
