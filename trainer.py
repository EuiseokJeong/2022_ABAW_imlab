import tensorflow as tf
import numpy as np
from utils import get_metric, get_loss
import time
from dataloader import dataloader


class Trainer():
    def __init__(self, teacher, student, alpha, beta, gamma, batch_size):
        self.ce = tf
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.teacher = teacher
        self.student = student
        self.t_train_loss, self.s_train_loss, self.t_valid_loss, self.s_valid_loss, self.t_valid_metric, self.s_valid_metric = [],[],[],[],[],[]
        self.t_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.)
        self.s_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.)
        self.train_dataloader = dataloader(type='train', batch_size=batch_size)
        self.valid_dataloader = dataloader(type='valid', batch_size=batch_size)
    def t_train_epoch(self, dataloader):
        # for epoch in range(self.epochs):
        loss_list = []
        iter = dataloader.max_iter
        dataloader.shuffle()
        st_time = time.time()
        for i, data in enumerate(dataloader):
            tmp = self.teacher.trainable_variables
            # print('\r', f"[INFO] ({i+1:0>5}/{iter:0>5}) Training teacher model || train_loss: {float(np.mean(loss_list)):.2f} {time.time()-st_time:.2f}sec", end = '')
            with tf.GradientTape() as tape:
                loss = 0
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    audios += 1
                    t_out = self.teacher(images, audios, training=True)
                    # tmp = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma)
                    loss += get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma)
            teacher_gradient = tape.gradient(loss, self.teacher.trainable_variables)
            self.t_optimizer.apply_gradients(zip(teacher_gradient, self.teacher.trainable_variables))
            loss_list.append(loss)
            print(f"[INFO] ({i + 1:0>5}/{iter:0>5}) Training teacher model || train_loss: {float(np.mean(loss_list)):.2f} {time.time() - st_time:.2f}sec")
        epoch_loss = float(np.mean(loss_list))
        return epoch_loss

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
    def train_teacher(self, epochs):
        best_metric = 0
        loss_list = []
        valid_metric_list = []
        for epoch in range(epochs):
            loss = self.t_train_epoch(self.train_dataloader)
            loss_list.append(loss)
            valid_metric = self.valid(self.valid_dataloader, self.teacher)
            valid_metric_list.append(valid_metric)
            if valid_metric > best_metric:
                self.save_weights()
            self.train_dataloader.shuffle()
            self.valid_dataloader.shuffle()
    def train_student(self, epochs):
        best_metric = 0
        loss_list = []
        valid_metric_list = []
        for epoch in range(epochs):
            loss = self.s_train_epoch(self.train_dataloader)
            loss_list.append(loss)
            valid_metric = self.valid(self.valid_dataloader, self.student)
            valid_metric_list.append(valid_metric)
            if valid_metric > best_metric:
                self.save_weights()
            self.train_dataloader.shuffle()
            self.valid_dataloader.shuffle()

    def save_weights(self):
        return
    def load_weights(self):
        return




    def valid(self, dataloader, model):
        iter = dataloader.max_iter
        st_time = time.time()
        valid_metric = []
        for i, data in enumerate(dataloader):
            for task_data in data:
                vid_names, idxes, images, audios, labels, task = task_data
                out = model(images, audios, training=False)
                # valid_loss[task].append((get_loss(out, labels, task, self.alpha, self.beta, self.gamma)))
                # valid_metric[task].append((get_metric(out, labels, task)))
                valid_metric.append(get_metric(out, labels, task))
                print(f"[INFO] ({i + 1:0>5}/{iter:0>5}) Validation model || valid_metric: {float(np.mean(valid_metric)):.2f} time: {time.time() - st_time:.2f}sec")

        return float(np.mean(valid_metric))
            # loss_list.append(loss)