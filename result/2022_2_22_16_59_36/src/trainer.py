import tensorflow as tf
import numpy as np
from utils import get_metric, get_loss, update_dict, check_dir, save_pickle
import time
from dataloader import dataloader
from config import configs
import os
import shutil
class Trainer():
    def __init__(self, teacher, student, alpha, beta, gamma, batch_size, gen):
        self.ce = tf
        self.alpha = alpha
        self.beta = beta
        self.gen = gen
        self.gamma = gamma
        self.teacher = teacher
        self.student = student
        self.t_train_loss, self.s_train_loss, self.t_valid_loss, self.s_valid_loss, self.t_valid_metric, self.s_valid_metric = [],[],[],[],[],[]
        self.t_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.)
        self.s_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.)
        self.train_dataloader = dataloader(type='train', batch_size=batch_size)
        self.valid_dataloader = dataloader(type='valid', batch_size=batch_size)
        self.set_result_path()
    def set_result_path(self):
        base_path = os.getcwd()
        result_path = os.path.join(base_path, 'result')
        check_dir(result_path)
        now = time.localtime()
        time_path = os.path.join(result_path,f"{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}")
        os.mkdir(time_path)
        self.weight_path = os.path.join(time_path, 'weight')
        self.plot_path = os.path.join(time_path, 'plot')
        src_path = os.path.join(time_path, 'src')
        self.dict_path = os.path.join(time_path, 'dict')
        os.mkdir(self.weight_path)
        os.mkdir(src_path)
        os.mkdir(self.plot_path)
        os.mkdir(self.dict_path)
        # save config in time_path
        shutil.copy(os.path.join(base_path, 'config.py'), os.path.join(time_path, 'saved_config.py'))
        # save source code in src_path
        py_list = [file for file in os.listdir(base_path) if file.endswith(".py")]
        for py in py_list:
            shutil.copy(os.path.join(base_path, py), os.path.join(src_path, py))

    def t_train_epoch(self, dataloader, T):
        # for epoch in range(self.epochs):
        loss_list = []
        train_loss = {'VA': [], 'EXPR': [], 'AU': []}
        iter = dataloader.max_iter
        dataloader.shuffle()
        st_time = time.time()
        for i, data in enumerate(dataloader):



            if i == 10:
                break



            with tf.GradientTape() as tape:
                loss = 0
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    t_out = self.teacher(images, audios, training=True)
                    task_loss = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, T)
                    loss += task_loss
                    train_loss[task].append(float(task_loss))
            teacher_gradient = tape.gradient(loss, self.teacher.trainable_weights)
            self.t_optimizer.apply_gradients(zip(teacher_gradient, self.teacher.trainable_weights))
            loss_list.append(float(loss))
            print('\r', f"[INFO] ({i + 1:0>5}/{iter:0>5}) Training teacher model || train_loss: {float(np.mean(loss_list)):.2f} ({time.time() - st_time:.1f}/{(time.time() - st_time)/(i+1)*iter:.1f})sec", end = '')
        for key in train_loss.keys():
            train_loss[key] = [float(np.mean(train_loss[key]))]
        return train_loss

    def s_train_epoch(self, dataloader):
        loss_list = []
        iter = dataloader.max_iter
        st_time = time.time()
        for i, data in enumerate(dataloader):
            with tf.GradientTape() as tape:
                loss = 0
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    t_out = self.teacher(images, audios, training=False)
                    s_out = self.student(images, audios, training=True)
                    loss += get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, s_out)
            student_gradient = tape.gradient(loss, self.student.trainable_variables)
            self.student.optimizer.apply_gradients(zip(student_gradient, self.student.trainable_variables))
            loss_list.append(loss)
            print(
                f"[INFO] ({i + 1:0>5}/{iter:0>5}) Training student model || train_loss: {float(np.mean(loss_list)):.2f} {time.time() - st_time:.2f}sec")
        epoch_loss = float(np.mean(loss_list))
        return epoch_loss
    def init_dict(self):
        self.train_loss_dict = {'VA': [], 'EXPR': [], 'AU': []}
        self.valid_metric_dict = {'VA': [], 'EXPR': [], 'AU': []}
        self.valid_loss_dict = {'VA': [], 'EXPR': [], 'AU': []}
    def train_teacher(self, epochs):
        best_metric = 0
        self.init_dict()
        early_stop = 0
        T = 1
        for epoch in range(epochs):
            train_loss_dict = self.t_train_epoch(self.train_dataloader, T)
            # loss_list.append(loss)
            valid_loss_dict, valid_metric_dict = self.valid(self.valid_dataloader, self.teacher, T)
            self.valid_loss_dict = update_dict(self.valid_loss_dict, valid_loss_dict)
            self.valid_metric_dict = update_dict(self.valid_metric_dict, valid_metric_dict)
            self.train_loss_dict = update_dict(self.train_loss_dict, train_loss_dict)
            mean_valid = float(np.mean([valid_metric_dict[x][0] for x in valid_metric_dict]))
            if mean_valid > best_metric:
                self.save_weights()
                early_stop = 0
            else:
                early_stop += 1
            if early_stop == configs['early_stop']:
                break
            self.train_dataloader.shuffle()
            self.valid_dataloader.shuffle()
            self.save_result()
    def save_result(self):
        # loss, metric 딕셔너리 저장
        save_pickle(self.train_loss_dict, os.path.join(self.dict_path, 'train_loss.pickle'))
        save_pickle(self.valid_loss_dict, os.path.join(self.dict_path, 'valid_loss.pickle'))
        save_pickle(self.valid_metric_dict, os.path.join(self.dict_path, 'valid_metric.pickle'))


    def train_student(self, epochs):
        best_metric = 0
        loss_list = []
        valid_metric_list = []
        for epoch in range(epochs):
            loss = self.s_train_epoch(self.train_dataloader)
            loss_list.append(loss)
            valid_loss, valid_metric = self.valid(self.valid_dataloader, self.student)
            valid_metric_list.append(valid_metric)
            if valid_metric > best_metric:
                self.save_weights()
            self.train_dataloader.shuffle()
            self.valid_dataloader.shuffle()

    def save_weights(self):
        return
    def load_weights(self):
        return

    def valid(self, dataloader, model, T):
        iter = dataloader.max_iter
        st_time = time.time()
        valid_metric = {'VA': [], 'EXPR':[], 'AU':[]}
        valid_loss = {'VA': [], 'EXPR': [], 'AU': []}
        for i, data in enumerate(dataloader):

            if i == 10:
                break


            loss = 0
            for task_data in data:
                vid_names, idxes, images, audios, labels, task = task_data
                out = model(images, audios, training=False)
                valid_loss[task].append(float(get_loss(out, labels, task, self.alpha, self.beta, self.gamma, T)))
                # valid_metric[task].append((get_metric(out, labels, task)))
                valid_metric[task].append(get_metric(out, labels, task))
                # valid_metric: {float(np.mean(valid_metric)):.2f}
                print(f"[INFO] ({i + 1:0>5}/{iter:0>5}) Validation model || time: {time.time() - st_time:.2f}sec")
        for key in valid_loss.keys():
            valid_loss[key] = [float(np.mean(valid_loss[key]))]
        for key in valid_metric.keys():
            valid_metric[key] = [float(np.mean(valid_metric[key]))]
        return valid_loss, valid_metric
            # loss_list.append(loss)