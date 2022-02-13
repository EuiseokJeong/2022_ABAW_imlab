import os
from moviepy.editor import VideoFileClip
from config import configs
from glob import glob
import cv2
from scipy.io import wavfile
import numpy as np

def video_to_audio(video_path, save_path):
    video_list = glob(os.path.join(video_path, '*'))
    for i, video_path in enumerate(video_list):
        video_name =video_path.split('/')[-1].split('.')[0]
        clip = VideoFileClip(video_path)

        clip.audio.write_audiofile(os.path.join(save_path, f"{video_name}.wav"), logger=None)
        print('\r', f"[INFO] ({i + 1}/{len(video_list)}) Converting video to wav {video_name}", end = '')

def audio_crop(video_path, audio_path, save_path):
    video_list = os.listdir(video_path)
    for i, video in enumerate(video_list):
        if video == '8-30-1280x720.mp4':
            print()
        print('\r', f"[INFO] ({i + 1}/{len(video_list)}) cropping wav file {video}", end='')
        save_dir = os.path.join(save_path, video.split(".")[0])
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        cap = cv2.VideoCapture(os.path.join(video_path, video))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        hz = cap.get(cv2.CAP_PROP_FPS)
        fs, data = wavfile.read(os.path.join(audio_path, f'{video.split(".")[0]}.wav'))
        idx_list = [(int((i+1)*(len(data)/frame_num)-44100/30), int((i+1)*(len(data)/frame_num))) for i in range(frame_num)]
        for j, idx in enumerate(idx_list):
            st_idx, end_idx = idx
            audio_chunk = data[st_idx:end_idx,:]
            wavfile.write(os.path.join(save_dir, f"{j}.wav"), 44100, audio_chunk.astype(np.int16))


if __name__ == '__main__':
    data_path = configs['data_path']
    print(data_path)
    video_path = os.path.join(data_path, 'video')
    audio_path = os.path.join(data_path, 'audio')
    cropped_audio_path = os.path.join(data_path, 'cropped_audio')
    if not os.path.isdir(audio_path):
        os.mkdir(audio_path)
    if not os.path.isdir(cropped_audio_path):
        os.mkdir(cropped_audio_path)
    # video_to_audio(video_path, audio_path)
    audio_crop(video_path, audio_path, cropped_audio_path)
