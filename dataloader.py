import numpy as np
import tensorflow as tf
import math
from skimage.io import imread
import os
import time
from glob import glob
import itertools
class dataloader(tf.keras.utils.Sequence):
    def __init__(self, type, batch_size, task_list = ['VA','EXPR','AU']):
        # task: 'VA', 'EXPR', 'AU', 'MTL
        self.batch_size = batch_size
        self.get_dict()
        self.data_path = os.path.join(os.getcwd(), 'data', '2022')
        self.get_idx(type, task_list)

    def __len__(self):
        return math.ceil(len(self.img_path_list) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.img_path_list[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.label_list[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        return np.array([imread(file_name) for file_name in batch_x]), np.array(batch_y)
    def get_idx(self, type, task_list):
        print("[INFO] Making index list")
        result_list = []
        img_path_list = []
        for task in task_list:
            task_vid_list = [x for x in os.listdir(os.path.join(self.data_path, 'annotation', self.task_dict[task], self.type_dict[type]))]
            result_list += [(x, task) for x in task_vid_list]
        for video, task in result_list:
            tmp_label = open(os.path.join(self.data_path, 'annotation', f'{self.task_dict[task]}', f'{self.type_dict[type]}', video), 'r', encoding='UTF8').read().splitlines()[1:]
            for i, label in enumerate(tmp_label):
                tmp_img_path = os.path.join(self.data_path, 'cropped_aligned', ''.join(video.split('.')[:-1]),  f"{i:0>5}.jpg")
                if '-1' in label.split(',') or '-5' in label.split(',') or not os.path.isfile(tmp_img_path):
                    continue
                img_path_list += [(i, tmp_img_path, self.convert_label(label, task), task)]
        self.img_path_list = [x[1] for x in img_path_list]
        self.idx_list = [x[0] for x in img_path_list]
        self.label_list = [x[2] for x in img_path_list]
        self.task_list = [x[3] for x in img_path_list]

    def convert_label(self, label, task):
        converted_label = np.zeros((22))
        if task == 'VA':
            converted_label[0:2] = np.array(label.split(','))
        elif task == 'EXPR':
            ont_hot_np = np.zeros((8))
            ont_hot_np[int(label)] = 1
            converted_label[2:10] = ont_hot_np
        elif task == 'AU':
            converted_label[10:] = np.array(label.split(','))
        else:
            raise ValueError(f"Task {task} is not valid!")
        return converted_label

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
    def n_images(self):
        return len(self.img_path_list)
if __name__ == '__main__':
    tmp = dataloader(type='train', batch_size=64)
    tot_time = time.time()
    st_time = time.time()
    for i , data in enumerate(tmp):
        print(i, st_time - time.time())
    print(f"total: {time.time()-tot_time}")