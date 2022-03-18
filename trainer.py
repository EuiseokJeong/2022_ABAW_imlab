import tensorflow as tf
import numpy as np
from utils import get_metric, get_loss, update_dict, check_dir, save_pickle, check_and_limit_gpu, check_weight
from models import get_model
import time
from dataloader import dataloader

import os
import shutil
import warnings
from silence_tensorflow import silence_tensorflow

class Trainer():
    def __init__(self, configs):
        self.configs = configs
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
        self.train_dataloader = dataloader(type='train', batch_size=configs['batch_size'], configs=configs)
        self.valid_dataloader = dataloader(type='valid', batch_size=configs['batch_size'], configs=configs)
        self.set_result_path()
        check_and_limit_gpu(self.configs['limit_gpu'])
    def set_result_path(self):
        base_path = os.getcwd()
        result_path = os.path.join(base_path, 'NAS', '2022', 'result')
        check_dir(result_path)
        now = time.localtime()
        self.time_path = os.path.join(result_path,f"{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}")
        os.mkdir(self.time_path)
        self.weight_path = os.path.join(self.time_path, 'weight')
        assert not os.path.isdir(self.weight_path), f"weight path {self.weight_path} already exists!"
        self.plot_path = os.path.join(self.time_path, 'plot')
        src_path = os.path.join(self.time_path, 'src')
        self.dict_path = os.path.join(self.time_path, 'dict')


        os.mkdir(self.weight_path)
        os.mkdir(src_path)
        os.mkdir(self.plot_path)
        os.mkdir(self.dict_path)
        # save config in time_path
        shutil.copyfile(os.path.join(base_path, 'config.py'), os.path.join(self.time_path, 'saved_config.py'))
        # save source code in src_path
        py_list = [file for file in os.listdir(base_path) if file.endswith(".py")]
        for py in py_list:
            shutil.copyfile(os.path.join(base_path, py), os.path.join(src_path, py))
    def refresh_path(self):
        # self.dict_path = os.path.join(self.dict_path, f"gen_{self.gen_cnt}")
        self.plot_path = os.path.join(self.plot_path, f"gen_{self.gen_cnt}")
        # self.weight_path = os.path.join(self.weight_path, f"gen_{self.gen_cnt}")
        for path in [self.plot_path]:
            check_dir(path)

    def init_dict(self):
        self.train_loss_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
        self.train_metric_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': []}
        self.valid_metric_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
        self.valid_loss_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
    def save_result(self):
        # loss, metric 딕셔너리 저장
        gen_dict = {
                'train_loss':self.train_loss_dict,
                'train_metric':self.train_metric_dict,
                'valid_loss':self.valid_loss_dict,
                'valid_metric':self.valid_metric_dict
        }
        save_pickle(gen_dict, os.path.join(self.dict_path, f'{self.gen_cnt}_result.pickle'))
    def write_result(self, valid_metric_dict, epoch):
        txt_path = os.path.join(self.time_path, 'train_valid.txt')
        f = open(txt_path, 'a')
        # if len(f.readlines) == 0:
        #     f.write('epoch, VA, EXPR, AU, MTL\n')
        valid_result = f'{epoch + 1}'
        for key in valid_metric_dict.keys():
            valid_result += f', {float(np.mean(valid_metric_dict[key]))}'
        valid_result += '\n'

        f.write(valid_result + '\n')
        f.close()

    def save_weights(self,epoch):
        path = os.path.join(self.weight_path, f'best_weight_gen_{self.gen_cnt}.h5')
        if self.gen_cnt == 0:
            self.teacher.save_weights(os.path.join(self.weight_path, f'best_weight_gen_{self.gen_cnt}.h5'), save_format='h5')
            self.teacher.save(os.path.join(self.weight_path, f'epoch({epoch+1})model_gen_{self.gen_cnt}'))
            del self.teacher
            self.teacher = tf.keras.models.load_model(os.path.join(self.weight_path, f'epoch({epoch+1})model_gen_{self.gen_cnt}'))
        else:
            self.student.save_weights(os.path.join(self.weight_path, f'best_weight_gen_{self.gen_cnt}.h5'),save_format='h5')
            self.student.save(os.path.join(self.weight_path, f'epoch({epoch + 1})model_gen_{self.gen_cnt}'))
            del self.student
            self.student = tf.keras.models.load_model(os.path.join(self.weight_path, f'epoch({epoch + 1})model_gen_{self.gen_cnt}'))
        print(f"            model saved in {os.path.join(self.weight_path, f'epoch({epoch+1})model_gen_{self.gen_cnt}')}")
    def load_model(self, gen_cnt):
        # self.gen_cnt -1은 가장 teacher
        # self.gen_cnt는 student
        model = get_model(self.configs)
        model(np.zeros((1, 6, 1, 512)), np.zeros((1, 1000)))

        model.load_weights(os.path.join(self.weight_path, f'best_weight_gen_{self.gen_cnt}.h5'))
        return model

    def train_gen(self):
        best_metric = 0
        self.init_dict()
        early_stop = 0
        self.valid_dataloader.shuffle()
        for epoch in range(self.epochs):

            self.train_dataloader.shuffle()
            train_loss_dict, train_metric_dict = self.train_epoch(self.train_dataloader, epoch)
            valid_loss_dict, valid_metric_dict = self.valid(self.teacher, self.valid_dataloader) if self.gen_cnt == 0 \
                else self.valid(self.student, self.valid_dataloader)
            self.write_result(valid_metric_dict, epoch)
            self.valid_loss_dict = update_dict(self.valid_loss_dict, valid_loss_dict)
            self.valid_metric_dict = update_dict(self.valid_metric_dict, valid_metric_dict)
            self.train_loss_dict = update_dict(self.train_loss_dict, train_loss_dict)
            self.train_metric_dict = update_dict(self.train_metric_dict, train_metric_dict)
            self.save_result()
            mean_valid = valid_metric_dict['MTL'][0]


            self.teacher.save_weights(os.path.join(self.weight_path, f"epoch({epoch+1})_MTL({mean_valid:.2f}).h5"))

            if mean_valid > best_metric:
                self.save_weights(epoch)
                early_stop = 0
                best_metric=mean_valid
            else:
                early_stop += 1
            if early_stop == self.configs['early_stop']:
                break
        self.plot_result()
    def plot_result(self):
        # self.train_loss_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': []}
        # self.train_metric_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': []}
        # self.valid_metric_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': []}
        # self.valid_loss_dict = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': []}
        return

    def train_epoch(self, dataloader, epoch):
        loss_list = []
        train_metric = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': []}
        train_loss = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
        iter = dataloader.max_iter
        st_time = time.time()
        T = 1 if self.gen_cnt == 0 else self.T
        t_training = True if self.gen_cnt == 0 else False
        for i, data in enumerate(dataloader):

            # if i == 10:
            #     break

            with tf.GradientTape() as tape:
                loss = 0
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    s_out = self.student([images, audios], training=True) if self.gen_cnt != 0 else None
                    t_out = self.teacher([images, audios], training=t_training)
                    task_loss = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, T, s_out=s_out)
                    task_metric = get_metric(t_out, labels, task, self.threshold) if self.gen_cnt == 0 else get_metric(s_out, labels, task, self.threshold)
                    if task_metric == 'nan':
                        continue

                    loss += task_loss
                    train_loss[task].append(float(task_loss))
                    train_metric[task].append(float(task_metric))
            if self.gen_cnt == 0:
                gradient = tape.gradient(loss, self.teacher.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.teacher.trainable_variables))
            else:
                gradient = tape.gradient(loss, self.student.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.student.trainable_variables))


                # loss = 0
                # for task_data in data:
                #     with tf.GradientTape() as tape:
                #         vid_names, idxes, images, audios, labels, task = task_data
                #         s_out = self.student([images, audios], training=True) if self.gen_cnt != 0 else None
                #         t_out = self.teacher([images, audios], training=t_training)
                #         task_loss = get_loss(t_out, labels, task, self.alpha, self.beta, self.gamma, T, s_out=s_out)
                #         task_metric = get_metric(t_out, labels, task, self.threshold) if self.gen_cnt == 0 else get_metric(s_out, labels, task, self.threshold)
                #         if task_metric == 'nan':
                #             continue
                #
                #     loss += task_loss
                #     train_loss[task].append(float(task_loss))
                #     train_metric[task].append(float(task_metric))
                #     if self.gen_cnt == 0:
                #         gradient = tape.gradient(loss, self.teacher.trainable_variables)
                #         self.optimizer.apply_gradients(zip(gradient, self.teacher.trainable_variables))
                #     else:
                #         gradient = tape.gradient(loss, self.student.trainable_variables)
                #         self.optimizer.apply_gradients(zip(gradient, self.student.trainable_variables))



            loss_list.append(float(loss))
            print('\r',f"[INFO] Gen_({self.gen_cnt}/{self.gen}) ({epoch+1}/{self.epochs})({i + 1:0>5}/{iter:0>5}) Training gen_{self.gen_cnt} model || train_loss: {float(np.mean(loss_list)):.2f} "
                       f"train_metric(VA/EXPR/AU/MTL): {float(np.mean(train_metric['VA'])):.2f}/{float(np.mean(train_metric['EXPR'])):.2f}/{float(np.mean(train_metric['AU'])):.2f}/{float(np.mean(train_metric['MTL'])):.2f} {time.time() - st_time:.2f}sec", end = '')
        for key in train_loss.keys():
            train_loss[key] = [float(np.mean(train_loss[key]))]
            train_metric[key] = [float(np.mean(train_metric[key]))]

        return train_loss, train_metric


    def valid(self, model, dataloader):
        print()
        iter = dataloader.max_iter
        dataloader.shuffle()
        st_time = time.time()
        valid_metric = {'VA': [], 'EXPR':[], 'AU':[], 'MTL':[]}
        valid_loss = {'VA': [], 'EXPR': [], 'AU': [], 'MTL':[]}
        for i, data in enumerate(dataloader):

            # if i == 10:
            #     break

            for task_data in data:
                vid_names, idxes, images, audios, labels, task = task_data
                # if self.gen_cnt == 0:
                #     out = self.teacher(images, audios, training=False)
                # else:
                #     out = self.student(images, audios, training=False)
                out = model([images, audios], training=False)
                valid_loss[task].append(float(get_loss(out, labels, task, self.alpha, self.beta, self.gamma, 1)))
                task_metric = get_metric(out, labels, task, self.threshold)
                if task_metric == 'nan':
                    continue
                valid_metric[task].append(task_metric)
                print('\r', f"      ({i + 1:0>5}/{iter:0>5}) Validation gen_{self.gen_cnt} model || valid_metric(VA/EXPR/AU/MTL): {float(np.mean(valid_metric['VA'])):.2f}/{float(np.mean(valid_metric['EXPR'])):.2f}/{float(np.mean(valid_metric['AU'])):.2f}/{float(np.mean(valid_metric['MTL'])):.2f} time: {time.time() - st_time:.2f}sec", end = '')
        for key in valid_loss.keys():
            valid_loss[key] = [float(np.mean(valid_loss[key]))]
        for key in valid_metric.keys():
            valid_metric[key] = [float(np.mean(valid_metric[key]))]
        print()
        return valid_loss, valid_metric
    def train(self):
        print(self.weight_path, self.configs, '\n')
        self.refresh_path()

        # imlab server
        # self.teacher = tf.keras.models.load_model('/home/euiseokjeong/Desktop/imlab/2022_ABAW_imlab/NAS/2022/result/keep/generation/2022_3_5_21_44_27(teacher_gen_0)/weight/epoch(28)model_gen_0')
        # 232
        self.teacher = tf.keras.models.load_model('/home/euiseokjeong/Desktop/IMLAB/ABAW/NAS/2022/result/keep/generation/2022_3_5_21_44_27(teacher_gen_0)/weight/epoch(28)model_gen_0')

        # self.teacher = get_model(self.configs)
        tf.keras.utils.plot_model(self.teacher, to_file=os.path.join(self.time_path, 'model.png'), show_shapes=True)
        # self.train_gen()

        for i in range(self.gen):
            self.gen_cnt += 1
            self.student = get_model(self.configs)
            self.train_gen()
if __name__=='__main__':
    from config import configs

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{configs['gpu_num']}"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    silence_tensorflow()
    warnings.filterwarnings(action='ignore')
    Trainer(configs).train()

    # trainer = Trainer(configs)
    # test_model = get_model(configs)
    # test_model(np.zeros((1,6,1,512)), np.zeros((1,1000)))
    # print('/home/euiseokjeong/Desktop/IMLAB/ABAW/result/keep/teacher_test/2022_2_25_20_22_32/weight/best_weight_gen_0.h5')
    # test_model.load_weights('/home/euiseokjeong/Desktop/IMLAB/ABAW/result/keep/teacher_test/2022_2_25_20_22_32/weight/best_weight_gen_0.h5')
    # trainer.valid(test_model, trainer.valid_dataloader)
