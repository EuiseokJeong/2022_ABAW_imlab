import tensorflow as tf
import numpy as np
from utils import get_metric, get_loss, update_dict, check_dir, save_pickle, check_and_limit_gpu, check_weight
from models import get_model
import time
from dataloader import dataloader
from config import configs
import os
import shutil
import warnings
from silence_tensorflow import silence_tensorflow

class Trainer():
    def __init__(self):
        self.gen_cnt = 0
        self.alpha = configs['alpha']
        self.epochs = configs['epochs']
        self.beta = configs['beta']
        self.gen = configs['generation']
        self.gamma = configs['beta']
        self.T = configs['temperature']
        self.t_train_loss, self.s_train_loss, self.t_valid_loss, self.s_valid_loss, self.t_valid_metric, self.s_valid_metric = [],[],[],[],[],[]
        self.t_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.)
        self.s_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.)
        self.train_dataloader = dataloader(type='train', batch_size=configs['batch_size'])
        self.valid_dataloader = dataloader(type='valid', batch_size=configs['batch_size'])
        self.set_result_path()
        self.train()
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
    def refresh_path(self):
        # self.dict_path = os.path.join(self.dict_path, f"gen_{self.gen_cnt}")
        self.plot_path = os.path.join(self.plot_path, f"gen_{self.gen_cnt}")
        # self.weight_path = os.path.join(self.weight_path, f"gen_{self.gen_cnt}")
        for path in [self.plot_path]:
            check_dir(path)
    def t_train_epoch(self, dataloader, epoch):
        loss_list = []
        train_loss = {'VA': [], 'EXPR': [], 'AU': []}
        iter = dataloader.max_iter
        st_time = time.time()
        for i, data in enumerate(dataloader):
            with tf.GradientTape() as tape:
                loss = 0
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    t_out = self.teacher(images, audios, training=True)
                    task_loss = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, 1)
                    loss += task_loss
                    train_loss[task].append(float(task_loss))
            teacher_gradient = tape.gradient(loss, self.teacher.trainable_weights)
            self.t_optimizer.apply_gradients(zip(teacher_gradient, self.teacher.trainable_weights))
            loss_list.append(float(loss))
            print('\r', f"[INFO] Gen_({self.gen_cnt}/{self.gen}) ({epoch+1}/{self.epochs})({i + 1:0>5}/{iter:0>5}) Training gen_{self.gen_cnt} model || train_loss: {float(np.mean(loss_list)):.2f} ({time.time() - st_time:.1f}/{(time.time() - st_time)/(i+1)*iter:.1f})sec", end = '')
        for key in train_loss.keys():
            train_loss[key] = [float(np.mean(train_loss[key]))]
        return train_loss

    def s_train_epoch(self, dataloader, epoch):
        loss_list = []
        train_loss = {'VA': [], 'EXPR': [], 'AU': []}
        iter = dataloader.max_iter
        st_time = time.time()
        for i, data in enumerate(dataloader):

            with tf.GradientTape() as tape:
                loss = 0
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    t_out = self.teacher(images, audios, training=False)
                    s_out = self.student(images, audios, training=True)
                    task_loss = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, self.T, s_out=s_out)
                    loss += task_loss
                    train_loss[task].append(float(task_loss))
            student_gradient = tape.gradient(loss, self.student.trainable_variables)
            self.s_optimizer.apply_gradients(zip(student_gradient, self.student.trainable_variables))
            loss_list.append(float(loss))
            print('\r',f"[INFO] Gen_({self.gen_cnt}/{self.gen}) ({epoch+1}/{self.epochs})({i + 1:0>5}/{iter:0>5}) Training gen_{self.gen_cnt} model || train_loss: {float(np.mean(loss_list)):.2f} {time.time() - st_time:.2f}sec", end = '')
        for key in train_loss.keys():
            train_loss[key] = [float(np.mean(train_loss[key]))]
        return train_loss
    def init_dict(self):
        self.train_loss_dict = {'VA': [], 'EXPR': [], 'AU': []}
        self.valid_metric_dict = {'VA': [], 'EXPR': [], 'AU': []}
        self.valid_loss_dict = {'VA': [], 'EXPR': [], 'AU': []}
    def train_teacher(self):
        best_metric = 0
        self.init_dict()
        early_stop = 0
        for epoch in range(self.epochs):
            train_loss_dict = self.t_train_epoch(self.train_dataloader, epoch)
            # loss_list.append(loss)
            valid_loss_dict, valid_metric_dict = self.valid(self.valid_dataloader, self.teacher, 1, epoch)
            self.valid_loss_dict = update_dict(self.valid_loss_dict, valid_loss_dict)
            self.valid_metric_dict = update_dict(self.valid_metric_dict, valid_metric_dict)
            self.train_loss_dict = update_dict(self.train_loss_dict, train_loss_dict)
            mean_valid = float(np.mean([valid_metric_dict[x][0] for x in valid_metric_dict]))
            self.save_result()
            if mean_valid > best_metric:
                self.save_weights()
                early_stop = 0
            else:
                early_stop += 1
            if early_stop == configs['early_stop']:
                break
            self.train_dataloader.shuffle()
            self.valid_dataloader.shuffle()
    def save_result(self):
        # loss, metric 딕셔너리 저장
        gen_dict = {
                'train_loss':self.train_loss_dict,
                'valid_loss':self.valid_loss_dict,
                'valid_metric':self.valid_metric_dict
        }
        save_pickle(gen_dict, os.path.join(self.dict_path, f'{self.gen_cnt}_result.pickle'))
        # save_pickle(self.valid_loss_dict, os.path.join(self.dict_path, f'{self.gen_cnt}_valid_loss.pickle'))
        # save_pickle(self.valid_metric_dict, os.path.join(self.dict_path, f'{self.gen_cnt}_valid_metric.pickle'))


    def train_student(self):
        best_metric = 0
        self.init_dict()
        early_stop = 0
        for epoch in range(self.epochs):
            train_loss_dict = self.s_train_epoch(self.train_dataloader, epoch)

            valid_loss_dict, valid_metric_dict = self.valid(self.valid_dataloader, self.student, self.T, epoch)
            self.valid_loss_dict = update_dict(self.valid_loss_dict, valid_loss_dict)
            self.valid_metric_dict = update_dict(self.valid_metric_dict, valid_metric_dict)
            self.train_loss_dict = update_dict(self.train_loss_dict, train_loss_dict)
            mean_valid = float(np.mean([valid_metric_dict[x][0] for x in valid_metric_dict]))
            self.save_result()

            if mean_valid > best_metric:
                # self.teacher.save_weights(os.path.join(self.weight_path, 'teacher.h5'))
                self.save_weights()
                early_stop = 0
            else:
                early_stop += 1
            if early_stop == configs['early_stop']:
                break
            self.train_dataloader.shuffle()
            self.valid_dataloader.shuffle()


            # loss_list.append(loss)
            # valid_loss, valid_metric = self.valid(self.valid_dataloader, self.student)
            # valid_metric_list.append(valid_metric)
            # if valid_metric > best_metric:
            #     self.save_weights()
            #     e
            # self.train_dataloader.shuffle()
            # self.valid_dataloader.shuffle()

    def save_weights(self):
        if self.gen_cnt==0:
            self.teacher.save_weights(os.path.join(self.weight_path, f't_gen{self.gen_cnt}.h5'))
        else:
            self.student.save_weights(os.path.join(self.weight_path, f's_gen{self.gen_cnt}.h5'))
    def load_weights(self):
        return

    def valid(self, dataloader, model, T, epoch):
        print()
        iter = dataloader.max_iter
        st_time = time.time()
        valid_metric = {'VA': [], 'EXPR':[], 'AU':[]}
        valid_loss = {'VA': [], 'EXPR': [], 'AU': []}
        for i, data in enumerate(dataloader):

            for task_data in data:
                vid_names, idxes, images, audios, labels, task = task_data
                out = model(images, audios, training=False)
                valid_loss[task].append(float(get_loss(out, labels, task, self.alpha, self.beta, self.gamma, T)))
                # valid_metric[task].append((get_metric(out, labels, task)))
                valid_metric[task].append(get_metric(out, labels, task))
                # valid_metric: {float(np.mean(valid_metric)):.2f}
                print('\r', f"      ({i + 1:0>5}/{iter:0>5}) Validation gen_{self.gen_cnt} model || valid_metric(VA/EXPR/AU): {float(np.mean(valid_loss['VA'])):.2f}/{float(np.mean(valid_loss['EXPR'])):.2f}/{float(np.mean(valid_loss['AU'])):.2f} time: {time.time() - st_time:.2f}sec", end = '')
        for key in valid_loss.keys():
            valid_loss[key] = [float(np.mean(valid_loss[key]))]
        for key in valid_metric.keys():
            valid_metric[key] = [float(np.mean(valid_metric[key]))]
        print()
        return valid_loss, valid_metric
            # loss_list.append(loss)
    def train(self):
        check_and_limit_gpu(configs['limit_gpu'])
        self.refresh_path()
        self.teacher = get_model(configs, teacher=True)
        self.train_teacher()
        print()
        for i in range(self.gen):
            self.gen_cnt += 1
            self.student = get_model(configs, teacher=False)
            self.train_student()
            print()

            self.teacher = get_model(configs, teacher=False)
            # self.teacher = self.teacher.build_graph()
            self.teacher(np.zeros((1,6,1,512)), np.zeros((1,1000)))
            self.teacher.set_weights(self.student.get_weights())
            # tmp = np.array(self.teacher.get_weights())
            # tmp2 = np.array(self.student.get_weights())
            check_weight(self.teacher, self.student)
            # assert (np.array(self.teacher.get_weights()) == np.array(self.student.get_weights())).all()
            # if (np.array(self.teacher.get_weights()) == np.array(self.student.get_weights())).all():
            #     raise ValueError("Cloning student model failed!")
if __name__=='__main__':
    silence_tensorflow()
    warnings.filterwarnings(action='ignore')
    Trainer()