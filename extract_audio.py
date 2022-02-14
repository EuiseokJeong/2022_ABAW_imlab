import os
from moviepy.editor import VideoFileClip
from config import configs
from glob import glob
import cv2
from scipy.io import wavfile
import numpy as np
import librosa
import soundfile as sf
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

def video_to_audio(video_path, save_path):
    video_list = glob(os.path.join(video_path, '*'))
    for i, video_path in enumerate(video_list):
        video_name =video_path.split('/')[-1].split('.')[0]
        clip = VideoFileClip(video_path)

        clip.audio.write_audiofile(os.path.join(save_path, f"{video_name}.wav"), logger=None)
        print('\r', f"[INFO] ({i + 1}/{len(video_list)}) Converting video to wav {video_name}", end = '')

def audio_crop(video_path, audio_path, save_path, sec = 10, sample_rate = 22050):
    video_list = os.listdir(video_path)
    for i, video in enumerate(video_list):
        print('\r', f"[INFO] ({i + 1}/{len(video_list)}) cropping wav file {video}", end='')
        save_dir = os.path.join(save_path, video.split(".")[0])
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        cap = cv2.VideoCapture(os.path.join(video_path, video))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data, sr = librosa.load(os.path.join(audio_path, f'{video.split(".")[0]}.wav'), sr = 22050)
        audio_per_frame = int(len(data) / frame_num)
        idx_list = [(i, audio_per_frame*i-sample_rate*sec, audio_per_frame*i) for i
                    in range(frame_num) if audio_per_frame*i-sample_rate*sec >=0]
        for j, idx in enumerate(idx_list):
            print('\r', f"[INFO] ({i + 1}/{len(video_list)}) cropping wav file {video} ({j/len(idx_list)*100:.1f}%)", end='')
            image_idx, st_idx, end_idx = idx
            audio_chunk = data[st_idx:end_idx]
            if len(audio_chunk) != sec*sample_rate:
                print(f'{os.path.join(save_dir, f"{image_idx}.wav")}')

            sf.write(os.path.join(save_dir, f"{image_idx}.wav"), audio_chunk, samplerate = sample_rate)
def extract_audio_feature(cropped_audio_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    soundnet = load_model(os.path.join('/home/euiseokjeong/Desktop/IMLAB/ABAW/', 'models', 'soundnet.hdf5'))
    video_list = os.listdir(cropped_audio_path)
    for i, video_name in enumerate(video_list):
        save_video_path = os.path.join(save_path, video_name)
        if not os.path.isdir(save_video_path):
            os.mkdir(save_video_path)
        file_path_list = glob(os.path.join(cropped_audio_path, video_name, '*'))
        for j, file_path in enumerate(file_path_list):
            print('\r',f"[INFO] ({i + 1}/{len(video_list)}) Extracting features from audio file {video_name} ({j / len(file_path_list) * 100:.1f}%)",end='')
            file_name = file_path.split('/')[-1].replace('.wav', '')
            x, sr = sf.read(file_path)
            # if the audio is shorter than 10 seconds, drop the samples
            if not len(x) >= sr * 10:
                continue
            feature = get_sound_features(soundnet, file_path)

            np.save(os.path.join(save_path, video_name, f"{file_name}.npy"), feature)

def get_sound_features(model, filepath):
    x, sr = sf.read(filepath)
    x = x[:10 * sr]

    x = x * 255.  # change range of x from -1~1 to -255.~255.
    x[x < -255.] = -255.  ## set the min saturation value to -255.
    x[x > 255.] = 255.  ## set the max saturation value to 255.
    x = np.reshape(x, (1, x.shape[0], 1, 1))  # reshape to (num_sample, length_audio, 1, 1)

    # chech the range of x
    assert np.max(x) <= 255., "It seems this audio contains signal that exceeds 256" + str(
        np.max(x)) + " : " + filepath
    assert np.min(x) >= -255., "It seems this audio contains signal that exceeds -256 " + str(
        np.min(x)) + " : " + filepath

    y_pred = model.predict(x)
    feature = y_pred[0][0][0][0]

    return feature



if __name__ == '__main__':
    data_path = configs['data_path']
    print(data_path)
    video_path = os.path.join(data_path, 'video')
    audio_path = os.path.join(data_path, 'audio')
    cropped_audio_path = os.path.join(data_path, 'cropped_audio')
    audio_feature_path = os.path.join(data_path, 'features', 'audio')
    if not os.path.isdir(audio_path):
        os.mkdir(audio_path)
    if not os.path.isdir(cropped_audio_path):
        os.mkdir(cropped_audio_path)
    # video_to_audio(video_path, audio_path)
    # audio_crop(video_path, audio_path, cropped_audio_path)
    extract_audio_feature(cropped_audio_path, audio_feature_path)
