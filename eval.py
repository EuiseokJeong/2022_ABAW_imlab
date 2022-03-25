import tensorflow as tf
import numpy as np
from utils import get_metric, check_dir, check_and_limit_gpu
import time
from dataloader import dataloader
from config import configs
import os
import warnings
from silence_tensorflow import silence_tensorflow

class tester():
    def __init__(self,configs):
        self.configs = configs
        self.result_path = configs['eval_path']
        print(self.result_path, configs)
    def valid(self):
        valid_dataloader = dataloader(type='valid', batch_size=self.configs['batch_size'], configs=self.configs)
        valid_dataloader.shuffle()
        iter = valid_dataloader.max_iter
        weight_path = os.path.join(self.result_path, 'weight')
        model_list = [x for x in os.listdir(weight_path) if len(x.split('.')) == 1]
        threshold = self.configs['au_threshold']
        assert len(model_list) != 0, f"weight_name is zero ({len(model_list)})"
        result_list = [f'model, VA, EXPR, AU, MTL, MTL/VA, MTL/EXPR, MTL/AU']
        for j, model_name in enumerate(model_list):
            st_time = time.time()
            valid_metric = {'VA': [], 'EXPR': [], 'AU': [], 'MTL': [], 'MTL/VA': [], 'MTL/EXPR': [], 'MTL/AU': []}
            test_model = tf.keras.models.load_model(os.path.join(weight_path, model_name))
            for i, data in enumerate(valid_dataloader):
                for task_data in data:
                    vid_names, idxes, images, audios, labels, task = task_data
                    out = test_model((images, audios), training=False)
                    task_metric,_ = get_metric(out, labels, task, threshold, get_per_task=True)
                    if task == 'MTL':
                        mtl_total, mtl_va, mtl_expr, mtl_au = task_metric
                        valid_metric['MTL'].append(mtl_total)
                        valid_metric['MTL/VA'].append(mtl_va)
                        valid_metric['MTL/EXPR'].append(mtl_expr)
                        valid_metric['MTL/AU'].append(mtl_au)
                    else:
                        valid_metric[task].append(task_metric)
                    if task_metric == 'nan':
                        continue
                    print('\r',f"[INFO] Validating model: ({j+1}/{len(model_list)}){model_name}, ({i + 1:0>5}/{iter:0>5}) || valid_metric(VA/EXPR/AU/MTL_total(va/expr/au)): {float(np.mean(valid_metric['VA'])):.2f}/{float(np.mean(valid_metric['EXPR'])):.2f}/{float(np.mean(valid_metric['AU'])):.2f}/{float(np.mean(valid_metric['MTL'])):.2f}({float(np.mean(valid_metric['MTL/VA'])):.2f}/{float(np.mean(valid_metric['MTL/EXPR'])):.2f}/{float(np.mean(valid_metric['MTL/AU'])):.2f}) time: {time.time() - st_time:.2f}sec",end='')
            print()

            weight_result = f'{model_name}'
            for key in valid_metric.keys():
                weight_result += f', {float(np.mean(valid_metric[key]))}'
            result_list.append(weight_result)
        result_path = os.path.join(self.result_path, 'valid_result.txt')
        result_list = [x+'\n' for x in result_list]
        with open(result_path, 'w') as f:
            f.writelines(result_list)


if __name__ == '__main__':
    silence_tensorflow()
    warnings.filterwarnings(action='ignore')
    check_and_limit_gpu(configs['limit_gpu'])
    print(f"\nresult_path: {configs['eval_path']}\n")
    tester = tester(configs)
    tester.valid()
