import cv2
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
import sys

class tester():
    def __init__(self,configs):
        self.configs = configs
        self.result_path = configs['eval_path']
        print(self.result_path, configs)

    # def load_model(self, model_path):
    #     # model = get_model(self.configs)
    #     # model(np.zeros((1,6,1,512)), np.zeros((1,1000)))
    #     # model.load_weights(weight_path)
    #     model = tf.keras.models.load_model('/home/euiseokjeong/Desktop/IMLAB/ABAW/result/2022_3_2_21_55_32/weight/epoch(10)model_gen_0')
    #     # /home/euiseokjeong/Desktop/IMLAB/ABAW/result/2022_3_2_21_55_32/weight/epoch(10)model_gen_0
    #     return model
    def valid(self):
        valid_dataloader = dataloader(type='valid', batch_size=self.configs['batch_size'], configs=self.configs)
        valid_dataloader.shuffle()
        iter = valid_dataloader.max_iter
        weight_path = os.path.join(self.result_path, 'weight')
        model_list = [x for x in os.listdir(weight_path) if len(x.split('.')) == 1]
        threshold = self.configs['au_threshold']
        assert len(model_list) != 0, f"weight_name is zero ({len(model_list)})"
        result_list = [f'model, VA, EXPR, AU, MTL']
        for j, model_name in enumerate(model_list):
            st_time = time.time()
            valid_metric = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': []}
            test_model = tf.keras.models.load_model(os.path.join(weight_path, model_name))
            for i, data in enumerate(valid_dataloader):
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    out = test_model((images, audios), training=False)
                    # valid_loss[task].append(float(get_loss(out, labels, task, self.alpha, self.beta, self.gamma, 1)))
                    # valid_metric[task].append((get_metric(out, labels, task)))
                    task_metric = get_metric(out, labels, task, threshold)
                    if task_metric == 'nan':
                        continue
                    valid_metric[task].append(task_metric)
                    # valid_metric: {float(np.mean(valid_metric)):.2f}
                    print('\r',f"[INFO] Validating model: ({j+1}/{len(model_list)}){model_name}, ({i + 1:0>5}/{iter:0>5}) || valid_metric(VA/EXPR/AU/MTL): {float(np.mean(valid_metric['VA'])):.2f}/{float(np.mean(valid_metric['EXPR'])):.2f}/{float(np.mean(valid_metric['AU'])):.2f}/{float(np.mean(valid_metric['MTL'])):.2f} time: {time.time() - st_time:.2f}sec",end='')
            print()

            weight_result = f'{model_name}'
            for key in valid_metric.keys():
                weight_result += f', {float(np.mean(valid_metric[key]))}'
            result_list.append(weight_result)
        result_path = os.path.join(self.result_path, 'valid_result.txt')
        result_list = [x+'\n' for x in result_list]
        with open(result_path, 'w') as f:
            f.writelines(result_list)
        # print()

                # valid_metric[key] = [float(np.mean(valid_metric[key]))]
    def write_submit(self, testset_list):
        test_path = os.path.join(self.result_path, 'test')
        check_dir(test_path)
        weight_path = os.path.join(self.result_path, 'weight')
        weight_list = os.listdir(weight_path)
        assert len(weight_list) != 0, f"weight_name is zero ({len(weight_list)})"
        for weight_name in weight_list:
            test_model = self.load_model(os.path.join(weight_path, weight_name))
            gen_path = os.path.join(test_path, weight_name)
            check_dir(gen_path)
            threshold = self.configs['au_threshold']
            for video_name in testset_list:
                video_path = [x for x in os.listdir(os.path.join(self.configs['data_path'], 'video')) if video_name in x.split('.')]
                assert len(video_path) != 0, f'video_name {video_name} is not in directory !'
                cap = cv2.VideoCapture(os.path.join(self.configs['data_path'], 'video', video_path[0]))
                frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prediction_list = ['image,valence,arousal,expression,aus']
                audio_path = os.path.join(self.configs['data_path'], 'features', 'audio',
                                          video_name.replace('_right', '').replace('_left', ''))
                img_path = os.path.join(self.configs['data_path'], 'features', 'image_t(2)_s(10)', video_name)
                audio_idx_list = [int(x.split('.')[0]) for x in os.listdir(audio_path)]
                img_idx_list = [int(x.split('.')[0]) for x in os.listdir(img_path)]
                for idx in range(frame_num):

                    if idx > 100:
                        continue

                    result_str = f"{video_name}/{idx+1:0>5}.jpg"
                    if idx not in audio_idx_list or idx not in img_idx_list:
                        prediction_list.append(result_str)
                        continue
                    audio_feature = np.expand_dims(np.load(os.path.join(audio_path, f"{idx}.npy")), axis=0)
                    img_feature = np.expand_dims(np.load(os.path.join(img_path, f"{idx}.npy")), axis=0)
                    # print(img_feature.shape)
                    _, va, expr, au = test_model(img_feature, audio_feature)
                    result = np.hstack([va, np.array([[np.argmax(expr)]]),
                                        np.where(np.where(au >= threshold, 1, au) < threshold, 0, np.where(au >= threshold, 1, au))])[0]
                    # result = np.hstack([va, np.argmax(expr), np.where(np.where(au >= threshold, 1, au) < threshold, 0, np.where(au >= threshold, 1, au))])
                    for x in result:
                        result_str += f',{x}'
                    prediction_list.append(result_str)
                prediction_list = self.postprocessing(prediction_list)
                prediction_list = [str(x)+'\n' for x in prediction_list]
                f = open(rf'{os.path.join(gen_path, f"{video_name}.txt")}', 'w')
                f.writelines(prediction_list)
                f.close()

    def postprocessing(self, prediction_list):
        for result in prediction_list:
            splitted = result.split(',')
        # st_flag = True
        # va: 처음에 없으면 -5, 중간에 없으면 interpolation, 끝에 없으면 이전 값 유지
        # expr: 처음에 없으면 -1, 중간에 없으면 이전 값 가져가기
        # au: 처음에 없으면 -1, 중간에 없으면 이전 값 가져가기
        # for x in prediction_list:
        #     if b
        return prediction_list








if __name__ == '__main__':
    silence_tensorflow()
    warnings.filterwarnings(action='ignore')
    check_and_limit_gpu(1024)
    # testset_list = ['video95']

    print(f"\nresult_path: {configs['eval_path']}\n")
    tester = tester(configs)
    # tester.write_submit(testset_list)
    tester.valid()
