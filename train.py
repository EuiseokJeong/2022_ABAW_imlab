import tensorflow as tf
import numpy as np

class Trainer():
    def __init__(self):
        pass
    def train_epoch(self, dataloader, teacher, student):
        for data in dataloader:
            with tf.GradientTape() as tape:
                loss = 0
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    t_out = teacher(images, audios, training=False)
                    s_out = student(images, audios, training=True)
                    loss += self.get_loss(t_out, s_out, labels, task)
            student_gradient = tape.gradient(loss, student.trainable_variables)
            student.optimizer.apply_gradients(zip(student_gradient, student.trainable_variables))

    def get_loss(self, t_out, s_out, labels, task):
        return 0