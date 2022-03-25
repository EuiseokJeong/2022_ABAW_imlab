import tensorflow as tf
import numpy as np
from utils import check_dir, check_and_limit_gpu
import time
from config import configs
import os
import warnings
from silence_tensorflow import silence_tensorflow

class evaluator():
    def __init__(self,configs):
        self.configs = configs
        self.result_path = configs['eval_path']
        self.data_path = configs['data_path']
        print(self.result_path, configs)

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
        test_num = len(testset_txt)

        for i, model_name in enumerate(model_list):
            f = open(os.path.join(submit_path, f"{model_name}.txt"), 'w')
            test_model = tf.keras.models.load_model(os.path.join(weight_path, model_name))
            st_time = time.time()
            result_list=['image,valence,arousal,expression,aus\n']
            for j, test_data in enumerate(testset_txt):
                print("\r", f'[INFO][{i+1}/{len(model_list)}({j+1}/{test_num})] making submission txt for {self.result_path.split("/")[-1]} time: {(j+1)/(time.time()-st_time):.2f}(it/s)',end='')
                image_name, image_num = test_data.split(',')[0].split('/')
                audio_name = image_name.replace('_left', '').replace('_right', '')
                image_num = int(image_num.replace('.jpg', ''))
                if not os.path.isdir(os.path.join(audio_path, audio_name)) or not os.path.isdir(os.path.join(image_path, image_name)):
                    tmp_result = ",".join([test_data.split(',')[0].replace('\n', '') + '.jpg'] + ['0' for x in range(22)]) + '\n'
                    result_list.append(tmp_result)
                    continue
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
            f.writelines(result_list)
            f.close()
        print()
    def get_near_num(self, img_path, aud_path, idx):
        img_list = [int(x.replace('.npy','')) for x in os.listdir(img_path)]
        aud_list = [int(x.replace('.npy', '')) for x in os.listdir(aud_path)]
        common_list= [x for x in img_list if x in aud_list]
        near_num = min(common_list, key=lambda x: abs(x - idx))
        return near_num

if __name__ == '__main__':
    silence_tensorflow()
    warnings.filterwarnings(action='ignore')
    check_and_limit_gpu(configs['limit_gpu'])
    print(f"\nresult_path: {configs['eval_path']}\n")
    tester = evaluator(configs).make_submit()
