import numpy as np
import tensorflow as tf
import math
from skimage.io import imread_collection
import os
import time
from config import configs
from scipy.io import wavfile
import gzip
import pickle
class dataloader(tf.keras.utils.Sequence):
    def __init__(self, type, batch_size, video_seq_len, audio_seq_len, task_list = ['VA','EXPR','AU']):
        # task: 'VA', 'EXPR', 'AU', 'MTL
        self.batch_size = batch_size
        self.get_dict()
        self.video_seq_len = video_seq_len
        self.audio_seq_len = audio_seq_len
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
        image_idx_list = [x[2] for x in data_list]
        audio_idx_list = [x[3] for x in data_list]
        task = data_list[0][5]
        label_np = np.array([x[4] for x in data_list])
        image_path_list = [[os.path.join(self.data_path, 'cropped_aligned', vid_name, f"{idx:0>5}.jpg") for idx in idx_list] for idx_list, vid_name in zip(image_idx_list, vid_name_list)]
        audio_path_list = [[os.path.join(self.data_path, 'cropped_audio', vid_name, f"{idx}.wav") for idx in idx_list] for idx_list, vid_name in zip(audio_idx_list, vid_name_list)]
        image_np = np.array([np.array(imread_collection(path_list)) for path_list in image_path_list])
        audio_np = np.array([np.array([wavfile.read(path)[1] for path in path_list]) for path_list in audio_path_list])
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
        idx_path = os.path.join(idx_dir_path, f'idx_{type}_vid({self.video_seq_len})_aud({self.audio_seq_len}).pickle')

        if os.path.isfile(idx_path):
            print(f"[INFO] Load existing idx pickle file in {idx_path}")
            # load and uncompress.
            st_time = time.time()
            with gzip.open(idx_path, 'rb') as f:
                self.idx_dict = pickle.load(f)
            print(f"load pickle file: {time.time() - st_time}")
        else:
            if not os.path.isdir(idx_dir_path):
                os.mkdir(idx_dir_path)
            self.idx_dict = {}
            for task in task_list:
                self.idx_dict[task] = []
                task_vid_list = [x for x in os.listdir(os.path.join(self.data_path, 'annotation', self.task_dict[task], self.type_dict[type]))]
                st_time = time.time()
                for k, video_name in enumerate(task_vid_list):
                    tmp_label = open(os.path.join(self.data_path, 'annotation', f'{self.task_dict[task]}', f'{self.type_dict[type]}',video_name), 'r', encoding='UTF8').read().splitlines()[1:]
                    video_name = video_name.replace('.txt', '')
                    audio_name = video_name.replace('.txt', '').replace('_left', '').replace('_right', '')
                    for i, label in enumerate(tmp_label):
                        print('\r',f"[INFO] Making index list task: {task} ({k + 1}/{len(task_vid_list)} {time.time() - st_time:.1f}sec)",end='')
                        img_idx = [n+1 for n in range(i - int(self.video_seq_len * 30), i, 1)]
                        audio_idx = [n+1 for n in range(i - int(self.audio_seq_len * 30), i, 1)]
                        if '-1' in label or '-5' in label or not self.check_path(img_idx, 'image', video_name) or not self.check_path(audio_idx, 'audio', audio_name):
                            continue
                        self.idx_dict[task].append((video_name, i, img_idx, audio_idx, self.convert_label(label, task), task))
            with gzip.open(idx_path, 'wb') as f:
                pickle.dump(self.idx_dict, f)
        self.va_list = self.idx_dict['VA']
        self.expr_list = self.idx_dict['EXPR']
        self.au_list = self.idx_dict['AU']
    def check_path(self, idx_list, stream, name):
        if stream == 'image':
            for idx in idx_list:
                if not os.path.isfile(os.path.join(self.data_path, 'cropped_aligned', name,  f"{idx:0>5}.jpg")):
                    return False
        elif stream == 'audio':
            for idx in idx_list:
                if not os.path.isfile(os.path.join(self.data_path, 'cropped_audio', name, f"{idx}.wav")):
                    return False
        else:
            raise ValueError(f"stream in self.check_path function {stream} is not valid!")
        return True

    def convert_label(self, label, task):
        if task == 'VA':
            return np.array(label.split(',')).astype(np.float32)
        elif task == 'EXPR':
            ont_hot_np = np.zeros((8))
            ont_hot_np[int(label)] = 1
            return ont_hot_np
        elif task == 'AU':
            return np.array(label.split(',')).astype(np.int8)
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
        pass
if __name__ == '__main__':
    # type, batch_size, video_seq_len, audio_seq_len, task_list = ['VA', 'EXPR', 'AU'], idx_path = None, save_idx = True)
    train_loader = dataloader(type='train', video_seq_len = 3, audio_seq_len=10, batch_size=64)
    tot_time = time.time()
    st_time = time.time()
    train_loader.shuffle()
    print(train_loader.max_iter)
    for i , data in enumerate(train_loader):
        print(i, time.time() - st_time)
        st_time = time.time()
    print(f"total: {time.time()-tot_time}")