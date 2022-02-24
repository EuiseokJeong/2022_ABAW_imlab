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
        self.gamma = configs['gamma']
        self.T = configs['temperature']
        self.threshold = configs['au_threshold']
        # self.t_train_loss, self.s_train_loss, self.t_valid_loss, self.s_valid_loss, self.t_valid_metric, self.s_valid_metric = [],[],[],[],[],[]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=configs['learning_rate'])
        self.train_dataloader = dataloader(type='train', batch_size=configs['batch_size'])
        self.valid_dataloader = dataloader(type='valid', batch_size=configs['batch_size'])
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
    def refresh_path(self):
        # self.dict_path = os.path.join(self.dict_path, f"gen_{self.gen_cnt}")
        self.plot_path = os.path.join(self.plot_path, f"gen_{self.gen_cnt}")
        # self.weight_path = os.path.join(self.weight_path, f"gen_{self.gen_cnt}")
        for path in [self.plot_path]:
            check_dir(path)

    def init_dict(self):
        self.train_loss_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
        self.valid_metric_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
        self.valid_loss_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
    def save_result(self):
        # loss, metric 딕셔너리 저장
        gen_dict = {
                'train_loss':self.train_loss_dict,
                'valid_loss':self.valid_loss_dict,
                'valid_metric':self.valid_metric_dict
        }
        save_pickle(gen_dict, os.path.join(self.dict_path, f'{self.gen_cnt}_result.pickle'))

    def save_weights(self, mean_valid):
        path = os.path.join(self.weight_path, f'gen_{self.gen_cnt}_val({mean_valid:.2f}).h5')
        if self.gen_cnt == 0:
            self.teacher.save_weights(path)
        else:
            self.student.save_weights(path)
        print(f"            weight saved in {path}")
    def load_weights(self):
        # self.gen_cnt -1은 가장 teacher
        # self.gen_cnt는 studeny
        return

    # def train_teacher(self):
    #     best_metric = 0
    #     self.init_dict()
    #     early_stop = 0
    #     for epoch in range(self.epochs):
    #         train_loss_dict = self.t_train_epoch(self.train_dataloader, epoch)
    #         # loss_list.append(loss)
    #         valid_loss_dict, valid_metric_dict = self.valid(self.valid_dataloader, self.teacher, 1, epoch)
    #         self.valid_loss_dict = update_dict(self.valid_loss_dict, valid_loss_dict)
    #         self.valid_metric_dict = update_dict(self.valid_metric_dict, valid_metric_dict)
    #         self.train_loss_dict = update_dict(self.train_loss_dict, train_loss_dict)
    #         tmp = [valid_metric_dict[x][0] for x in valid_metric_dict]
    #         mean_valid = float(np.mean([valid_metric_dict[x][0] for x in valid_metric_dict]))
    #         self.save_result()
    #         if mean_valid > best_metric:
    #             self.save_weights()
    #             early_stop = 0
    #             best_metric = mean_valid
    #             print(f'               Weight saved! mean valid: {mean_valid}')
    #         else:
    #             early_stop += 1
    #         if early_stop == configs['early_stop']:
    #             break
    #         self.train_dataloader.shuffle()
    #         self.valid_dataloader.shuffle()
    def train_gen(self):
        best_metric = 0
        self.init_dict()
        early_stop = 0
        for epoch in range(self.epochs):
            train_loss_dict = self.train_epoch(self.train_dataloader, epoch)

            if self.gen_cnt == 0:
                valid_loss_dict, valid_metric_dict = self.valid(self.valid_dataloader, self.teacher, 1)
            else:
                valid_loss_dict, valid_metric_dict = self.valid(self.valid_dataloader, self.student, self.T)
            self.valid_loss_dict = update_dict(self.valid_loss_dict, valid_loss_dict)
            self.valid_metric_dict = update_dict(self.valid_metric_dict, valid_metric_dict)
            self.train_loss_dict = update_dict(self.train_loss_dict, train_loss_dict)
            mean_valid = float(np.mean([valid_metric_dict[x][0] for x in valid_metric_dict]))
            self.save_result()
            if mean_valid > best_metric:
                # self.teacher.save_weights(os.path.join(self.weight_path, 'teacher.h5'))
                self.save_weights(mean_valid)
                early_stop = 0
                best_metric=mean_valid
            else:
                early_stop += 1
            if early_stop == configs['early_stop']:
                break
            self.train_dataloader.shuffle()
            self.valid_dataloader.shuffle()
    # def train_student(self):
    #     best_metric = 0
    #     self.init_dict()
    #     early_stop = 0
    #     for epoch in range(self.epochs):
    #         train_loss_dict = self.s_train_epoch(self.train_dataloader, epoch)
    #
    #         valid_loss_dict, valid_metric_dict = self.valid(self.valid_dataloader, self.student, self.T, epoch)
    #         self.valid_loss_dict = update_dict(self.valid_loss_dict, valid_loss_dict)
    #         self.valid_metric_dict = update_dict(self.valid_metric_dict, valid_metric_dict)
    #         self.train_loss_dict = update_dict(self.train_loss_dict, train_loss_dict)
    #         mean_valid = float(np.mean([valid_metric_dict[x][0] for x in valid_metric_dict]))
    #         self.save_result()
    #
    #         if mean_valid > best_metric:
    #             # self.teacher.save_weights(os.path.join(self.weight_path, 'teacher.h5'))
    #             self.save_weights()
    #             early_stop = 0
    #             best_metric=mean_valid
    #         else:
    #             early_stop += 1
    #         if early_stop == configs['early_stop']:
    #             break
    #         self.train_dataloader.shuffle()
    #         self.valid_dataloader.shuffle()
    # def t_train_epoch(self, dataloader, epoch):
    #     loss_list = []
    #     train_loss = {'VA': [], 'EXPR': [], 'AU': []}
    #     iter = dataloader.max_iter
    #     st_time = time.time()
    #     for i, data in enumerate(dataloader):
    #         with tf.GradientTape() as tape:
    #             loss = 0
    #             for task_data in data:
    #                 vid_names, idxes, images, audios, labels, task = task_data
    #                 t_out = self.teacher(images, audios, training=True)
    #                 task_loss = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, 1)
    #                 loss += task_loss
    #                 train_loss[task].append(float(task_loss))
    #         teacher_gradient = tape.gradient(loss, self.teacher.trainable_weights)
    #         self.t_optimizer.apply_gradients(zip(teacher_gradient, self.teacher.trainable_weights))
    #         loss_list.append(float(loss))
    #         print('\r', f"[INFO] Gen_({self.gen_cnt}/{self.gen}) ({epoch+1}/{self.epochs})({i + 1:0>5}/{iter:0>5}) Training gen_{self.gen_cnt} model || train_loss: {float(np.mean(loss_list)):.2f} ({time.time() - st_time:.1f}/{(time.time() - st_time)/(i+1)*iter:.1f})sec", end = '')
    #     for key in train_loss.keys():
    #         train_loss[key] = [float(np.mean(train_loss[key]))]
    #     return train_loss
    #
    # def s_train_epoch(self, dataloader, epoch):
    #     loss_list = []
    #     train_loss = {'VA': [], 'EXPR': [], 'AU': []}
    #     iter = dataloader.max_iter
    #     st_time = time.time()
    #     for i, data in enumerate(dataloader):
    #         with tf.GradientTape() as tape:
    #             loss = 0
    #             for task_data in data:
    #                 vid_names, idxes, images, audios, labels, task = task_data
    #                 t_out = self.teacher(images, audios, training=False)
    #                 s_out = self.student(images, audios, training=True)
    #                 task_loss = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, self.T, s_out=s_out)
    #                 loss += task_loss
    #                 train_loss[task].append(float(task_loss))
    #         student_gradient = tape.gradient(loss, self.student.trainable_variables)
    #         self.s_optimizer.apply_gradients(zip(student_gradient, self.student.trainable_variables))
    #         loss_list.append(float(loss))
    #         print('\r',
    #               f"[INFO] Gen_({self.gen_cnt}/{self.gen}) ({epoch + 1}/{self.epochs})({i + 1:0>5}/{iter:0>5}) Training gen_{self.gen_cnt} model || train_loss: {float(np.mean(loss_list)):.2f} {time.time() - st_time:.2f}sec",
    #               end='')
    #     for key in train_loss.keys():
    #         train_loss[key] = [float(np.mean(train_loss[key]))]
    #     return train_loss
    def train_epoch(self, dataloader, epoch):
        loss_list = []
        train_loss = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
        iter = dataloader.max_iter
        st_time = time.time()
        if self.gen_cnt == 0:
            s_out = None
            T = 1
            t_training = True
        else:
            T = self.T
            t_training = False
        for i, data in enumerate(dataloader):

            with tf.GradientTape() as tape:
                loss = 0
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    if self.gen_cnt != 0:
                        s_out = self.student(images, audios, training=True)
                    t_out = self.teacher(images, audios, training=t_training)
                    task_loss = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, T, s_out=s_out)
                    loss += task_loss
                    train_loss[task].append(float(task_loss))
            if self.gen_cnt == 0:
                gradient = tape.gradient(loss, self.teacher.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.teacher.trainable_variables))
            else:
                gradient = tape.gradient(loss, self.student.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.student.trainable_variables))
            loss_list.append(float(loss))
            print('\r',f"[INFO] Gen_({self.gen_cnt}/{self.gen}) ({epoch+1}/{self.epochs})({i + 1:0>5}/{iter:0>5}) Training gen_{self.gen_cnt} model || train_loss: {float(np.mean(loss_list)):.2f} {time.time() - st_time:.2f}sec", end = '')
        for key in train_loss.keys():
            train_loss[key] = [float(np.mean(train_loss[key]))]
        return train_loss


    def valid(self, dataloader, model, T):
        print()
        iter = dataloader.max_iter
        st_time = time.time()
        valid_metric = {'VA': [], 'EXPR':[], 'AU':[], 'MTL':[]}
        valid_loss = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
        for i, data in enumerate(dataloader):

            for task_data in data:
                vid_names, idxes, images, audios, labels, task = task_data
                out = model(images, audios, training=False)
                valid_loss[task].append(float(get_loss(out, labels, task, self.alpha, self.beta, self.gamma, T)))
                # valid_metric[task].append((get_metric(out, labels, task)))
                valid_metric[task].append(get_metric(out, labels, task, self.threshold))
                # valid_metric: {float(np.mean(valid_metric)):.2f}
                print('\r', f"      ({i + 1:0>5}/{iter:0>5}) Validation gen_{self.gen_cnt} model || valid_metric(VA/EXPR/AU): {float(np.mean(valid_metric['VA'])):.2f}/{float(np.mean(valid_metric['EXPR'])):.2f}/{float(np.mean(valid_metric['AU'])):.2f} time: {time.time() - st_time:.2f}sec", end = '')
        for key in valid_loss.keys():
            valid_loss[key] = [float(np.mean(valid_loss[key]))]
        for key in valid_metric.keys():
            valid_metric[key] = [float(np.mean(valid_metric[key]))]
        print()
        return valid_loss, valid_metric
    def train(self):
        check_and_limit_gpu(configs['limit_gpu'])
        print(self.weight_path, configs, '\n')
        self.refresh_path()
        self.teacher = get_model(configs)
        self.train_gen()
        for i in range(self.gen):
            print()
            self.gen_cnt += 1
            self.student = get_model(configs)
            self.train_gen()
            print()
            del self.teacher
            self.teacher = get_model(configs)
            self.teacher(np.zeros((1,6,1,512)), np.zeros((1,1000)))
            self.teacher.set_weights(self.student.get_weights())
            check_weight(self.teacher, self.student)
            del self.student
if __name__=='__main__':
    silence_tensorflow()
    warnings.filterwarnings(action='ignore')
    Trainer().train()