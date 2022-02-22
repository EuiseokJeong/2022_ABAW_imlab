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
from utils import check_and_limit_gpu, check_dir
from models.FER_model.ResNet import ResNet34

def video_to_audio(video_path, save_path):
    video_list = glob(os.path.join(video_path, '*'))
    for i, video_path in enumerate(video_list):
        video_name =video_path.split('/')[-1].split('.')[0]
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(os.path.join(save_path, f"{video_name}.wav"), logger=None)
        print('\r', f"[INFO] ({i + 1}/{len(video_list)}) Converting video to wav {video_name}", end = '')

def audio_crop(video_path, audio_path, save_path, sec = 10, sample_rate = 22050):
    video_list = os.listdir(video_path)
    video_list = [x for x in video_list if 'mp4' in x.split('.') or 'avi' in x.split('.')]
    for i, video in enumerate(video_list):
        print('\r', f"[INFO] ({i + 1}/{len(video_list)}) cropping wav file {video}", end='')
        save_dir = os.path.join(save_path, video.split(".")[0])
        if os.path.isdir(save_dir):
            continue
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        cap = cv2.VideoCapture(os.path.join(video_path, video))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data, sr = librosa.load(os.path.join(audio_path, f'{video.split(".")[0]}.wav'), sr = 22050)
        if len(data) < sec*sample_rate:
            print(f"shorter than 10 sec: {video}")
            continue
        audio_per_frame = int(len(data) / frame_num)
        idx_list = [(i, audio_per_frame*i-sample_rate*sec, audio_per_frame*i) for i
                    in range(frame_num) if audio_per_frame*i-sample_rate*sec >=0]
        for j, idx in enumerate(idx_list):
            try:
                image_idx, st_idx, end_idx = idx
                wav_path = os.path.join(save_dir, f"{image_idx}.wav")
                print('\r', f"[INFO] ({i + 1}/{len(video_list)}) cropping wav file {video} ({j / len(idx_list) * 100:.1f}%)", end='')
                if os.path.isfile(wav_path):
                    continue
                audio_chunk = data[st_idx:end_idx]
                if len(audio_chunk) != sec*sample_rate:
                    print(f'{os.path.join(save_dir, f"{image_idx}.wav")}')
                sf.write(wav_path, audio_chunk, samplerate = sample_rate)
            except:
                print(video, idx, wav_path)
def extract_audio_feature(cropped_audio_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    soundnet = load_model(os.path.join(os.getcwd(), 'models', 'soundnet.hdf5'))
    video_list = os.listdir(cropped_audio_path)
    for i, video_name in enumerate(video_list):
        file_path_list = glob(os.path.join(cropped_audio_path, video_name, '*'))
        if len(file_path_list) == 0:
            print(video_name)
            continue
        save_video_path = os.path.join(save_path, video_name)
        check_dir(save_video_path)
        st_time = time.time()
        for j, file_path in enumerate(file_path_list):
            print('\r',f"[INFO] ({i + 1}/{len(video_list)}) Extracting features from audio file {video_name} [{j}/{len(file_path_list)}({j / len(file_path_list) * 100:.2f}%)] time: {time.time() - st_time:.1f}s",end='')
            file_name = file_path.split('/')[-1].replace('.wav', '')
            if os.path.isfile(os.path.join(save_path, video_name, f"{file_name}.npy")):
                continue
            x, sr = sf.read(file_path)
            # if the audio is shorter than 10 seconds, drop the samples
            if not len(x) >= sr * 10:
                continue
            feature = get_sound_features(soundnet, file_path)
            np.save(os.path.join(save_path, video_name, f"{file_name}.npy"), feature)
        print()

def extract_image_feature(cropped_image_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    video_list = os.listdir(cropped_image_path)
    fer_tuned_model = load_capnet_model()
    for i, video_name in enumerate(video_list):
        file_path_list = glob(os.path.join(cropped_image_path, video_name, '*'))
        if len(file_path_list) == 0:
            print(video_name)
            continue
        save_video_path = os.path.join(save_path, video_name)
        check_dir(save_video_path)
        st_time = time.time()
        for j, file_path in enumerate(file_path_list):
            file_name = file_path.split('/')[-1].replace('.jpg', '')
            print('\r', f"[INFO] ({i + 1}/{len(video_list)}) Extracting features from image file {video_name} [{j}/{len(file_path_list)}({j / len(file_path_list) * 100:.2f}%)] time: {time.time() - st_time:.1f}s", end='')
            if os.path.isfile(os.path.join(save_path, video_name, f"{file_name}.npy")):
                continue
            feature = get_image_feature(fer_tuned_model, file_path)
            # np.save(os.path.join(save_path, video_name, f"{file_name}.npy"), feature)
        print()

def merge_image_feature(image_feature_path, save_path, time_window, stride, minus_num):
    video_list = os.listdir(image_feature_path)
    for i, video in enumerate(video_list):
        st_time = time.time()
        video_save_path = os.path.join(save_path, video)
        check_dir(video_save_path)
        feature_list = os.listdir(os.path.join(image_feature_path, video))
        idx_list = [int(x.replace('.npy', '')) for x in feature_list]
        for j, (idx, feature) in enumerate(zip(idx_list, feature_list)):
            print('\r',
                  f"[INFO] ({i + 1}/{len(video_list)}) Merging image features from file {video} [{j}/{len(idx_list)}({j / len(idx_list) * 100:.2f}%)] time: {time.time() - st_time:.1f}s",
                  end='')
            if os.path.isfile(os.path.join(video_save_path, f'{idx}.npy')):
                continue
            result = True
            idx_range = range(idx-30*time_window+stride, idx+stride, stride)
            for x in idx_range:
                if x not in idx_list:
                    result = False
            if result:
                merged_feature = np.array([np.load(os.path.join(image_feature_path, video, f'{x:0>5}.npy')) for x in idx_range])
                np.save(os.path.join(video_save_path, f'{idx}.npy'), merged_feature)
        minus_name(video_save_path, 'npy', minus_num=minus_num)
        print()


    return

def minus_name(path, extension, minus_num = -1):
    file_list = os.listdir(path)
    file_list = [int(x.replace(f'.{extension}','')) for x in file_list if extension in x.split('.')]
    file_list.sort()
    for file in file_list:
#         file_name = int(file.split('.')[0])
        source = os.path.join(path, f'{file}.{extension}')
        dest = os.path.join(path, f'{file+minus_num}.{extension}')
        os.rename(source, dest)

def get_sound_features(model, filepath):
    x, sr = sf.read(filepath)
    x = x[:10 * sr]
    x = x * 255.  # change range of x from -1~1 to -255.~255.
    x[x < -255.] = -255.  ## set the min saturation value to -255.
    x[x > 255.] = 255.  ## set the max saturation value to 255.
    x = np.reshape(x, (1, x.shape[0], 1, 1))  # reshape to (num_sample, length_audio, 1, 1)
    assert np.max(x) <= 255., "It seems this audio contains signal that exceeds 256" + str(
        np.max(x)) + " : " + filepath
    assert np.min(x) >= -255., "It seems this audio contains signal that exceeds -256 " + str(
        np.min(x)) + " : " + filepath
    y_pred = model.predict(x)
    feature = y_pred[0][0][0][0]
    return feature

def get_image_feature(model, filepath) :
    img_raw = tf.io.read_file(filepath)
    img_rgb = tf.image.decode_jpeg(img_raw, channels=3)
    img_rs = tf.image.resize(img_rgb, [224, 224])   # reshape to (224, 224, 3)
    img_norm = img_rs / 255.    # change range of x from 0~225. to 0~1
    x = tf.expand_dims(img_norm, axis=0)    # expand dimension for batch size
    feature = model.predict(x)
    return feature

def load_capnet_model():
    # load feature extractor of CAPNet from the scripts in the model directory
    base_model = ResNet34(cardinality=32, se='parallel_add')
    # set the weights
    path_base_weights = os.path.join(os.getcwd(), 'models', 'weights', 'best_weights')
    base_model.load_weights(path_base_weights)
    # build base model
    base_model.build(input_shape=(None, 224, 224, 3))
    # split the feature extractor
    feature_extractor = tf.keras.Sequential()
    feature_extractor.add(tf.keras.Input(shape=(224, 224, 3)))
    for i in range(6):
        feature_extractor.add(base_model.layers[i])
    return feature_extractor

if __name__ == '__main__':


    check_and_limit_gpu(configs['limit_gpu'])
    data_path = configs['data_path']
    stride = configs['stride']
    time_win = configs['time_window']

    video_path = os.path.join(data_path, 'video')
    audio_path = os.path.join(data_path, 'audio')
    cropped_audio_path = os.path.join(data_path, 'cropped_audio')
    cropped_image_path = os.path.join(data_path, 'cropped_aligned')
    audio_feature_path = os.path.join(data_path, 'features', 'audio')
    image_feature_path = os.path.join(data_path, 'features', 'singe_image_feature')
    merge_path = os.path.join(data_path, 'features', f'image_t({time_win})_s({stride})')
    check_dir(audio_path)
    check_dir(cropped_audio_path)
    check_dir(merge_path)

    # video_to_audio(video_path, audio_path)
    # audio_crop(video_path, audio_path, cropped_audio_path)
    extract_audio_feature(cropped_audio_path, audio_feature_path)
    # extract_image_feature(cropped_image_path, image_feature_path)
    # merge_image_feature(image_feature_path, merge_path, time_window=time_win, stride=stride, minus_num = 0)
