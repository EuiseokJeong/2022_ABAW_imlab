import numpy as np
import tensorflow as tf
import os
import time
from sklearn.metrics import f1_score
from config import configs
from tensorflow.keras.layers import Softmax
import pickle
import gzip

def check_and_limit_gpu(memory_limit = 1024 * 3):
    print()
    ################### Limit GPU Memory ###################
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")
    # set the only one GPU and memory limit
    memory_limit = memory_limit
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU is not available')
    print()

def CCC_score(x, y):
    x_mean = tf.math.reduce_mean(x)
    y_mean = tf.math.reduce_mean(y)
    x_var = tf.math.reduce_variance(x)
    y_var = tf.math.reduce_variance(y)

    sxy = tf.math.reduce_mean((x - x_mean) * (y - y_mean))

    ccc = (2 * sxy) / (x_var + y_var + tf.math.pow(x_mean - y_mean, 2))
    return ccc

# Loss function
def loss_ccc(x, y) :
    items = [CCC_score(x[:, 0], y[:, 0]), CCC_score(x[:, 1], y[:, 1])]
    total_ccc = tf.math.reduce_mean(items)
    loss = 1 - total_ccc
    return loss

# Metric function
def metric_CCC(x, y):
    cccs = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    return cccs

def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def f1_metric(x, y):
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average='macro')
    return float(np.mean(f1))
def get_metric(out, labels, task):
    t_f, t_va, t_expr, t_au = out
    if task == 'VA':
        ccc = metric_CCC(t_va, labels)
        metric = float(np.mean(ccc))
    elif task == 'EXPR':
        metric = f1_metric(t_expr, labels)
    elif task == 'AU':
        metric = f1_metric(t_au, labels)
    else:
        raise ValueError(f"Task {task} is not valid!")
    return metric
# class custom_loss():
#     def __init__(self, alpha, beta, gamma, mmd):
#         self.ce_loss = tf.keras.losses.CategoricalCrossentropy()
#         self.bce_loss = tf.keras.losses.BinaryCrossentropy()
#     def get_loss(self, t_out, labels, task, s_out=None):
#
#     def va_loss(self, t_out, labels, task, s_out=None):
#         s_f, s_va, s_expr, s_au = s_out
#         if s_out is not None:
#
#
# t_out, labels, task, self.alpha, self.beta, self.gamma, s_out
def check_weight(src_model, target_model):
    src_weights = src_model.get_weights()
    target_weights = target_model.get_weights()
    for src_weight, target_weight in zip(src_weights, target_weights):
        assert (src_weight == target_weight).all()

def get_loss(t_out, labels, task, alpha, beta, gamma, T, s_out=None, mmd=False):
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    softmax = Softmax()
    t_f, t_va, t_expr, t_au = t_out
    if s_out is not None:
        s_f, s_va, s_expr, s_au = s_out
        t_expr = softmax(t_expr/T)
        s_expr = softmax(s_expr/T)


        loss = 0
        if task == 'VA':
            # tmp = alpha * loss_ccc(s_va, labels)
            # tmp2 = loss_ccc(s_va, t_va)
            # tmp3 = beta * cce_loss(t_expr, s_expr)
            # tmp4 = bce_loss(t_au, s_au)
            # tmp5 = beta * (cce_loss(t_expr, s_expr) + bce_loss(t_au, s_au))
            # tmp6 = tmp + tmp2 + tmp3 + tmp4 + tmp5

            # tmp_ = loss_ccc(s_va, labels)
            # loss += alpha * loss_ccc(s_va, labels)
            # loss += loss_ccc(s_va, t_va)
            # tmp = (cce_loss(t_expr, s_expr))
            # loss += beta * tmp
            # loss += bce_loss(t_au, s_au)
            loss = alpha * loss_ccc(s_va, labels) + loss_ccc(s_va, t_va) + \
                   beta * (cce_loss(t_expr, s_expr) + bce_loss(t_au, s_au))
        elif task == 'EXPR':
            # tmp_7 = cce_loss(s_expr, labels)
            # tmp = alpha * cce_loss(s_expr, labels)
            # tmp2 = cce_loss(s_expr, t_expr)
            # tmp3 = loss_ccc(s_va, t_va)
            # tmp4 = bce_loss(t_au, s_au)
            # # tmp5 = beta * (cce_loss(t_expr, s_expr) + bce_loss(t_au, s_au))
            # # tmp6 = tmp + tmp2 + tmp3 + tmp4 + tmp5
            # tmp_ = cce_loss(s_expr, labels)
            # loss += alpha * cce_loss(s_expr, labels)
            # tmp = cce_loss(s_expr, t_expr)
            # loss += tmp
            # loss += loss_ccc(s_va, t_va)
            # loss += bce_loss(t_au, s_au)


            loss = alpha * cce_loss(s_expr, labels) + cce_loss(s_expr, t_expr) + \
                   beta * (loss_ccc(s_va, t_va) + bce_loss(t_au, s_au))
        elif task == 'AU':
            # tmp = alpha * cce_loss(s_expr, labels)
            # tmp2 = cce_loss(s_expr, t_expr)
            # tmp3 = beta * (loss_ccc(s_va, t_va))
            # tmp4 = bce_loss(t_au, s_au)

            loss = alpha * bce_loss(s_au, labels) + bce_loss(s_au, t_au) + \
                   beta * (loss_ccc(s_va, t_va) + cce_loss(s_expr, t_expr))
        else:
            raise ValueError(f"Task {task} is not valid!")
        if mmd:
            loss += gamma * mmd(t_f, s_f)
    elif s_out is None:
        t_expr = softmax(t_expr / T)
        if task == 'VA':
            loss = loss_ccc(t_va, labels)
        elif task == 'EXPR':
            loss = cce_loss(t_expr, labels)
        elif task == 'AU':
            loss = bce_loss(t_au, labels)
        else:
            raise ValueError(f"Task {task} is not valid!")
    # print(task, loss)
    return tf.cast(loss, dtype=tf.float64)

def update_dict(prev_dict, add_dict):
    for key in add_dict.keys():
        assert len(add_dict[key]) == 1, f"length of data in key({key}) is not one! ({len(add_dict[key])})"
        if key in prev_dict.keys():
            prev_dict[key].append(add_dict[key][0])
    return prev_dict

def get_result_path(path):
    result_path = os.path.join(path, 'result')
    check_dir(result_path)
    now = time.localtime()
    time_path = os.path.join(result_path,
                             f"{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}")
    os.mkdir(time_path)
    weight_path = os.path.join(time_path, 'weight')
    src_path = os.path.join(time_path, 'src')
    os.mkdir(weight_path)
    os.mkdir(src_path)
    return time_path, weight_path, src_path
def save_pickle(dict, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(dict, f)
def load_pickle(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data
