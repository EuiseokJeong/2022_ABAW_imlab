import cv2
import tensorflow as tf
import numpy as np
from utils import get_metric, check_dir, check_and_limit_gpu
import time
from dataloader import dataloader
from config import configs
import os
import warnings
from silence_tensorflow import silence_tensorflow

class evaluator():
    def __init__(self,configs):
        self.configs = configs
        self.result_path = configs['eval_path']
        self.data_path = configs['data_path']
        # /home/euiseokjeong/Desktop/IMLAB/2022_ABAW_imlab/NAS/2022/testset/Multi_Task_Learning_Challenge_test_set_release.txt
        print(self.result_path, configs)

    # def load_model(self, model_path):
    #     # model = get_model(self.configs)
    #     # model(np.zeros((1,6,1,512)), np.zeros((1,1000)))
    #     # model.load_weights(weight_path)
    #     model = tf.keras.models.load_model('/home/euiseokjeong/Desktop/IMLAB/ABAW/result/2022_3_2_21_55_32/weight/epoch(10)model_gen_0')
    #     # /home/euiseokjeong/Desktop/IMLAB/ABAW/result/2022_3_2_21_55_32/weight/epoch(10)model_gen_0
    #     return model
    def make_submit(self):
        softmax = tf.keras.layers.Softmax()
        testset_path = os.path.join(self.data_path, 'testset', 'Multi_Task_Learning_Challenge_test_set_release.txt')
        submit_path = os.path.join(self.result_path, 'submission')
        check_dir(submit_path)
        f = open(testset_path,'r')
        testset_txt = f.readlines()
        testset_txt = [x.split(',')[0].replace('.jpg', '') for x in testset_txt[1:]]

        image_path = os.path.join(self.data_path, 'features', 'image_t(2)_s(10)')
        audio_path = os.path.join(self.data_path, 'features', 'audio')

        weight_path = os.path.join(self.result_path, 'weight')
        model_list = [x for x in os.listdir(weight_path) if len(x.split('.')) == 1]
        threshold = self.configs['au_threshold']
        assert len(model_list) != 0, f"The number of model is zero ({len(model_list)})"

        test_num = len(testset_txt[1:])
        for i, model_name in enumerate(model_list):

            f = open(os.path.join(submit_path, f"{model_name}.txt"), 'w')

            test_model = tf.keras.models.load_model(os.path.join(weight_path, model_name))
            result_list=['image,valence,arousal,expression,aus\n']
            for j, test_data in enumerate(testset_txt[1:]):

                if j ==10:
                    break


                print("\r", f'[INFO] ({j+1}/{test_num}) making submission txt for {self.result_path.split("/")[-1]}',end='')
                image_name, image_num = test_data.split(',')[0].split('/')
                audio_name = image_name.replace('_left', '').replace('_right', '')
                image_num = int(image_num.replace('.jpg', ''))
                image_feature_path = os.path.join(image_path, image_name, f"{image_num}.npy")
                audio_feature_path = os.path.join(audio_path, audio_name, f"{image_num}.npy")
                if not os.path.isfile(image_feature_path) or not os.path.isfile(audio_feature_path):
                    image_num = self.get_near_num(os.path.join(image_path, image_name),os.path.join(audio_path, audio_name),image_num)
                    image_feature_path = os.path.join(image_path, image_name, f"{image_num}.npy")
                    audio_feature_path = os.path.join(audio_path, audio_name, f"{image_num}.npy")
                img_feature = np.load(image_feature_path)
                aud_feature = np.load(audio_feature_path)
                result = test_model([np.expand_dims(img_feature, axis=0), np.expand_dims(aud_feature, axis=0)])
                va_out = list(np.array(result[1])[0])
                expr_out = [np.argmax(softmax(result[2]))]
                au_out = list(((np.array(result[3]) >= threshold) * 1)[0])
                tmp_result = ",".join([test_data.split(',')[0].replace('\n','')+'.jpg'] + [str(x) for x in (va_out + expr_out + au_out)]) + '\n'
                result_list.append(tmp_result)

             # save result_list as save to "{model_name}.txt"
            f.writelines(result_list)
            f.close()



    def get_near_num(self, img_path, aud_path, idx):
        img_list = [int(x.replace('.npy','')) for x in os.listdir(img_path)]
        aud_list = [int(x.replace('.npy', '')) for x in os.listdir(aud_path)]
        common_list= [x for x in img_list if x in aud_list]
        near_num = min(common_list, key=lambda x: abs(x - idx))
        return near_num
        # result_list = [f'model, VA, EXPR, AU, MTL, MTL/VA, MTL/EXPR, MTL/AU']
        # for j, model_name in enumerate(model_list):
        #     st_time = time.time()
        #     valid_metric = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': [], 'MTL/VA': [], 'MTL/EXPR': [], 'MTL/AU': []}
        #     test_model = tf.keras.models.load_model(os.path.join(weight_path, model_name))
        #     for i, data in enumerate(valid_dataloader):
        #         for task_data in data:
        #             vid_names, idxes, images, audios, labels, task = task_data
        #             out = test_model((images, audios), training=False)
        #             # valid_loss[task].append(float(get_loss(out, labels, task, self.alpha, self.beta, self.gamma, 1)))
        #             # valid_metric[task].append((get_metric(out, labels, task)))
        #             task_metric,_ = get_metric(out, labels, task, threshold, get_per_task=True)
        #             if task == 'MTL':
        #                 mtl_total, mtl_va, mtl_expr, mtl_au = task_metric
        #                 valid_metric['MTL'].append(mtl_total)
        #                 valid_metric['MTL/VA'].append(mtl_va)
        #                 valid_metric['MTL/EXPR'].append(mtl_expr)
        #                 valid_metric['MTL/AU'].append(mtl_au)
        #             else:
        #                 valid_metric[task].append(task_metric)
        #             if task_metric == 'nan':
        #                 continue
        #             print('\r',f"[INFO] Validating model: ({j+1}/{len(model_list)}){model_name}, ({i + 1:0>5}/{iter:0>5}) || valid_metric(VA/EXPR/AU/MTL_total(va/expr/au)): {float(np.mean(valid_metric['VA'])):.2f}/{float(np.mean(valid_metric['EXPR'])):.2f}/{float(np.mean(valid_metric['AU'])):.2f}/{float(np.mean(valid_metric['MTL'])):.2f}({float(np.mean(valid_metric['MTL/VA'])):.2f}/{float(np.mean(valid_metric['MTL/EXPR'])):.2f}/{float(np.mean(valid_metric['MTL/AU'])):.2f}) time: {time.time() - st_time:.2f}sec",end='')
        #     print()
        #
        #     weight_result = f'{model_name}'
        #     for key in valid_metric.keys():
        #         weight_result += f', {float(np.mean(valid_metric[key]))}'
        #     result_list.append(weight_result)
        # result_path = os.path.join(self.result_path, 'valid_result.txt')
        # result_list = [x+'\n' for x in result_list]
        # with open(result_path, 'w') as f:
        #     f.writelines(result_list)

if __name__ == '__main__':
    silence_tensorflow()
    warnings.filterwarnings(action='ignore')
    check_and_limit_gpu(configs['limit_gpu'])
    # testset_list = ['video95']

    print(f"\nresult_path: {configs['eval_path']}\n")
    tester = evaluator(configs)
    # tester.write_submit(testset_list)
    tester.make_submit()
