import numpy as np
import tensorflow as tf
import os
import time
from sklearn.metrics import f1_score
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

def loss_ccc(x, y) :
    items = [CCC_score(x[:, 0], y[:, 0]), CCC_score(x[:, 1], y[:, 1])]
    total_ccc = tf.math.reduce_mean(items)
    loss = 1 - total_ccc
    return loss

def metric_CCC(x, y):
    cccs = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    return cccs

def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def expr_f1_metric(x, y):
    x = Softmax()(x)
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
    f1 = f1_score(x, y, average='macro', zero_division=1)
    return float(np.mean(f1))

def au_f1_metric(input, target, threshold):
    input = (np.array(input) >= threshold)*1
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i],zero_division=1)
        f1s.append(f1)
    return np.mean(f1s)

def get_metric(out, labels, task, au_threshold, get_per_task=False):
    t_domain, t_va, t_expr, t_au = out
    if task == 'MTL':
        label = np.stack([np.hstack(x) for x in labels])
        va_l = label[:, 0:2]
        expr_l = label[:, 2:10]
        au_l = label[:, 10:]
        ccc = metric_CCC(t_va, va_l)
        mtl_va = float(np.mean(ccc))
        mtl_expr = expr_f1_metric(t_expr, expr_l)
        mtl_au = au_f1_metric(t_au, au_l, au_threshold)
        task_metric = mtl_va + mtl_expr + mtl_au
        if get_per_task:
            return (task_metric, mtl_va, mtl_expr, mtl_au),None
        return task_metric, None
    elif task == 'VA':
        ccc = metric_CCC(t_va, labels)
        task_metric = float(np.mean(ccc))
    elif task == 'EXPR':
        task_metric = expr_f1_metric(t_expr, labels)
    elif task == 'AU':
        task_metric = au_f1_metric(t_au, labels, au_threshold)
    else:
        raise ValueError(f"Task {task} is not valid!")
    if np.isnan(task_metric).any():
        return 'nan'
    domain_label = get_domain_label(task, labels.shape[0])
    domain_metric = tf.keras.metrics.CategoricalAccuracy()(t_domain, domain_label)
    return task_metric, domain_metric
<<<<<<< HEAD

=======
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
>>>>>>> 13a1dd7f55e75618c82af3e3a17162f882ad0024
def check_weight(src_model, target_model):
    src_weights = src_model.get_weights()
    target_weights = target_model.get_weights()
    for src_weight, target_weight in zip(src_weights, target_weights):
        assert (src_weight == target_weight).all()

def get_loss(t_out, labels, task, alpha, beta, domain_weight, T, non_improve_list, task_weight, exp = None, s_out=None):
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    softmax = Softmax(dtype='float32')
    t_domain, t_va, t_expr, t_au = t_out
    if s_out is not None:
        s_domain, s_va, s_expr, s_au = s_out
        t_expr = softmax(t_expr/T)
        s_expr = softmax(s_expr/T)
        task_weight_dict = {'VA':1, 'EXPR':1,'AU':1,'MTL':1}
        if task_weight:
            task_weight_dict = {}
            for key in non_improve_list:
                task_weight_dict[key] = np.exp(exp * non_improve_list[key])
        if task == 'MTL':
            label = np.stack([np.hstack(x) for x in labels])
            va_l = label[:, 0:2]
            expr_l = label[:, 2:10]
            au_l = label[:, 10:]
            total_loss = task_weight_dict['VA'] * loss_ccc(s_va, va_l) + task_weight_dict['EXPR'] * cce_loss(s_expr, expr_l) + task_weight_dict['AU'] * bce_loss(s_au, au_l)
            return total_loss

        elif task == 'VA':
<<<<<<< HEAD
=======
            tmp = alpha * task_weight_dict['VA'] * loss_ccc(s_va, labels)
            tmp2 = task_weight_dict['VA']*loss_ccc(s_va, t_va)
            tmp3 = beta * (task_weight_dict['EXPR'] * cce_loss(t_expr, s_expr))
            tmp4 = task_weight_dict['AU']*bce_loss(t_au, s_au)
            tmp5 = tmp3+tmp4
>>>>>>> 13a1dd7f55e75618c82af3e3a17162f882ad0024
            task_loss = alpha * task_weight_dict['VA'] * loss_ccc(s_va, labels) + task_weight_dict['VA']*loss_ccc(s_va, t_va) + \
                   beta * (task_weight_dict['EXPR'] * cce_loss(t_expr, s_expr) + task_weight_dict['AU']*bce_loss(t_au, s_au))
        elif task == 'EXPR':
            task_loss = alpha * task_weight_dict['EXPR'] * cce_loss(s_expr, labels) + task_weight_dict['EXPR'] * cce_loss(s_expr, t_expr) + \
                   beta * (task_weight_dict['VA'] * loss_ccc(s_va, t_va) + task_weight_dict['AU'] * bce_loss(t_au, s_au))
        elif task == 'AU':
            task_loss = alpha * task_weight_dict['AU'] * bce_loss(s_au, labels) + task_weight_dict['AU'] * bce_loss(s_au, t_au) + \
                    beta * (task_weight_dict['VA'] * loss_ccc(s_va, t_va) + task_weight_dict['EXPR'] * cce_loss(s_expr, t_expr))
        domain_label = get_domain_label(task, labels.shape[0])
        domain_loss = cce_loss(s_domain, domain_label)
        total_loss = task_loss + domain_weight * domain_loss

    elif s_out is None:
        t_expr = softmax(t_expr / T)
        if task == 'MTL':
            label = np.stack([np.hstack(x) for x in labels])
            va_l = label[:, 0:2]
            expr_l = label[:, 2:10]
            au_l = label[:, 10:]
            task_loss = loss_ccc(t_va, va_l) + cce_loss(t_expr, expr_l) + bce_loss(t_au, au_l)
            return task_loss

        elif task == 'VA':
            task_loss = loss_ccc(t_va, labels)
        elif task == 'EXPR':
            task_loss = cce_loss(t_expr, labels)
        elif task == 'AU':
            task_loss = bce_loss(t_au, labels)
        else:
            raise ValueError(f"Task {task} is not valid!")
        domain_label = get_domain_label(task, labels.shape[0])
        domain_loss = cce_loss(t_domain, domain_label)
        total_loss = task_loss + domain_weight * domain_loss
    return total_loss

def get_domain_label(task, num):
    domain_dict = {
        'VA': [1, 0, 0],
        'EXPR': [0, 1, 0],
        'AU': [0, 0, 1],
    }
    return np.array([domain_dict[task] for _ in range(num)], dtype=np.float32)

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
