import sys
import numpy as np
import librosa
import tensorflow as tf

model1="best_emotion_model.keras"
sr=22050
n_mels=128
max_len=175

emotions = ["Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"]

def extract_log_mel(y,sr):
    mel=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128)
    log_mel=librosa.power_to_db(mel,ref=np.max)
    return log_mel

def pad_spectrogram(spec, max_len=175):
    if spec.shape[1] < max_len:
        pad_width = max_len - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :max_len]
    return spec

def preprocess_audio(file_path):
    y, _ = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y)

    spec = extract_log_mel(y, sr)
    spec = pad_spectrogram(spec)

    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
    spec = spec[..., np.newaxis]  # (128, 175, 1)
    spec = np.expand_dims(spec, axis=0)  # (1, 128, 175, 1)

    return spec


if len(sys.argv) != 2:
    print("Usage: python predict.py <audio_path.wav>")
    sys.exit(1)
audio_path=sys.argv[1]

model = tf.keras.models.load_model(model1)

X = preprocess_audio(audio_path)
probs = model.predict(X)[0]

pred_idx = np.argmax(probs)
confidence = probs[pred_idx] * 100

print(f"Predicted Emotion: {emotions[pred_idx]}")
print(f"Confidence: {confidence:.2f}%")
