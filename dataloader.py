import numpy as np
import tensorflow as tf
import math
from skimage.io import imread_collection
import os
import time
from config import configs
from scipy.io import wavfile
import gzip
import librosa
import random
import pickle
class dataloader(tf.keras.utils.Sequence):
    def __init__(self, type, batch_size, video_seq_len=2, stride = 10, task_list = ['VA','EXPR','AU']):
        # task: 'VA', 'EXPR', 'AU', 'MTL
        self.batch_size = batch_size
        self.stride = stride
        self.get_dict()
        self.video_seq_len = video_seq_len
        self.data_path = configs['data_path']
        self.get_idx(type, task_list)
        self.get_batch_size()


    def __len__(self):
        return math.ceil(self.max_len / self.batch_size)

    def __getitem__(self, idx):
        batch_va_list = self.va_list[idx * self.va_batch_size:(idx + 1) * self.va_batch_size]
        batch_expr_list = self.expr_list[idx * self.expr_batch_size:(idx + 1) * self.expr_batch_size]
        batch_au_list = self.au_list[idx * self.au_batch_size:(idx + 1) * self.au_batch_size]
        return self.get_data(batch_va_list), self.get_data(batch_expr_list), self.get_data(batch_au_list)
    def get_data(self, data_list):
        vid_name_list = [x[0] for x in data_list]
        idx_list = [x[1] for x in data_list]
        label_np = np.array([x[2] for x in data_list])
        task = data_list[0][3]

        image_np = np.array([np.load(os.path.join(self.img_feature_path, vid_name, f"{idx}.npy")) for idx, vid_name in zip(idx_list, vid_name_list)])
        audio_np = np.array([np.load(os.path.join(self.audio_feature_path, vid_name.replace('_left', '').replace('_right', ''), f"{idx}.npy")) for idx, vid_name in zip(idx_list, vid_name_list)])

        return vid_name_list, idx_list, image_np, audio_np, label_np, task

    def get_batch_size(self):
        len_list = []
        for key in self.idx_dict.keys():
            len_list.append(len(self.idx_dict[key]))
        self.max_len = max(len_list)
        self.max_iter = math.ceil(self.max_len / self.batch_size)
        self.va_batch_size = int(len(self.idx_dict['VA']) / self.max_iter)
        self.expr_batch_size = int(len(self.idx_dict['EXPR']) / self.max_iter)
        self.au_batch_size = int(len(self.idx_dict['AU']) / self.max_iter)


    def get_idx(self, type, task_list):
        idx_dir_path = os.path.join(self.data_path, 'idx')
        idx_path = os.path.join(idx_dir_path, f'idx_{type}_vid({self.video_seq_len})_str({self.stride}).pickle')
        self.img_feature_path = os.path.join(self.data_path, 'features', f'image_t({self.video_seq_len})_s({self.stride})')
        self.audio_feature_path = os.path.join(self.data_path, 'features', 'audio')
        if os.path.isfile(idx_path):
            print(f"[INFO] Load existing idx pickle file in {idx_path}")
            # load and uncompress.
            st_time = time.time()
            with gzip.open(idx_path, 'rb') as f:
                self.idx_dict = pickle.load(f)
            # print(f"load pickle file: {time.time() - st_time}")
        else:
            if not os.path.isdir(idx_dir_path):
                os.mkdir(idx_dir_path)
            self.idx_dict = {}
            for task in task_list:
                self.idx_dict[task] = []
                task_vid_list = os.listdir(os.path.join(self.data_path, 'annotation', self.task_dict[task], self.type_dict[type]))
                st_time = time.time()
                for k, video_name in enumerate(task_vid_list):
                    tmp_label = open(os.path.join(self.data_path, 'annotation', f'{self.task_dict[task]}', f'{self.type_dict[type]}',video_name), 'r', encoding='UTF8').read().splitlines()[1:]
                    video_name = video_name.replace('.txt', '')
                    audio_name = video_name.replace('_left', '').replace('_right', '')
                    audio_path = os.path.join(self.audio_feature_path, audio_name)
                    image_path = os.path.join(self.img_feature_path, video_name)
                    if not os.path.isdir(audio_path) or not os.path.isdir(image_path):
                        continue
                    audio_idx_list = [int(x.replace('.npy', '')) for x in os.listdir(audio_path) if 'npy' in x.split('.')]
                    image_idx_list = [int(x.replace('.npy', '')) for x in os.listdir(image_path) if 'npy' in x.split('.')]
                    for idx, label in enumerate(tmp_label):
                        print('\r',f"[INFO] Making index list task: {task} ({k + 1}/{len(task_vid_list)} {time.time() - st_time:.1f}sec)",end='')
                        if '-1' in label or '-5' in label or idx not in audio_idx_list or idx not in image_idx_list:
                            continue
                        self.idx_dict[task].append((video_name, idx, self.convert_label(label, task), task))
                print()
            with gzip.open(idx_path, 'wb') as f:
                pickle.dump(self.idx_dict, f)
        self.va_list = self.idx_dict['VA']
        self.expr_list = self.idx_dict['EXPR']
        self.au_list = self.idx_dict['AU']

    def convert_label(self, label, task):
        if task == 'VA':
            return np.array(label.split(',')).astype(np.float32)
        elif task == 'EXPR':
            ont_hot_np = np.zeros((8))
            ont_hot_np[int(label)] = 1
            return ont_hot_np
        elif task == 'AU':
            return np.array(label.split(',')).astype(np.float32)
        else:
            raise ValueError(f"Task {task} is not valid!")
        # return converted_label

    def get_dict(self):
        self.type_dict = {
            'train': 'Train_Set',
            'valid': 'Validation_Set'
        }
        self.task_dict = {
            'VA': 'VA_Estimation_Challenge',
            'EXPR': 'EXPR_Classification_Challenge',
            'AU': 'AU_Detection_Challenge'
        }

    def shuffle(self):
        # 딕셔너리  섞어주고 다시 각각 리스트 만들어주
        for key in self.idx_dict.keys():
            self.idx_dict[key] = random.shuffle(self.idx_dict[key])
        pass
if __name__ == '__main__':
    # check_and_limit_gpu(configs['limit_gpu'])
    data_path = configs['data_path']
    stride = configs['stride']
    time_win = configs['time_window']
    train_loader = dataloader(type='train', video_seq_len = time_win, stride = stride, batch_size=128)
    tot_time = time.time()
    st_time = time.time()

    print(train_loader.max_iter)
    for i , data in enumerate(train_loader):
        vid_names, idxes, images, audios, labels, task = data[0]
        print(i, time.time() - st_time)
        st_time = time.time()
    print(f"total: {time.time()-tot_time}")
    train_loader.shuffle()