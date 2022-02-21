import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import f1_score

def check_and_limit_gpu(memory_limit = 1024 * 3):
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

def get_loss(t_out, labels, task, alpha, beta, gamma, s_out=None, mmd=False):
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    t_f, t_va, t_expr, t_au = t_out
    if s_out is not None:
        s_f, s_va, s_expr, s_au = s_out
        if task == 'VA':
            loss = alpha * loss_ccc(s_va, labels) + loss_ccc(s_va, t_va) + \
                   beta * (cce_loss(t_expr, s_expr) + cce_loss(t_au, s_au))
        elif task == 'EXPR':
            loss = alpha * cce_loss(s_expr, labels) + cce_loss(s_expr, t_expr) + \
                   beta * (loss_ccc(s_va, t_va) + cce_loss(t_au, s_au))
        elif task == 'AU':
            loss = alpha * cce_loss(s_au, labels) + cce_loss(s_au, t_au) + \
                   beta * (loss_ccc(s_va, t_va) + cce_loss(s_expr, t_expr))
        else:
            raise ValueError(f"Task {task} is not valid!")
        if mmd:
            loss += gamma * mmd(t_f, s_f)
    elif s_out is None:
        if task == 'VA':
            loss = loss_ccc(t_va, labels)
        elif task == 'EXPR':
            loss = cce_loss(t_expr, labels)
        elif task == 'AU':
            loss = cce_loss(t_au, labels)
        else:
            raise ValueError(f"Task {task} is not valid!")
    return tf.cast(loss, dtype=tf.float32)
