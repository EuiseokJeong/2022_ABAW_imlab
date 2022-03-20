configs = {
    # setting
    'stride':10,
    'time_window':2,
    'data_path': '/home/euiseokjeong/Desktop/IMLAB/2022_ABAW_imlab/data/2022',
    'limit_gpu':1024*3,
    'gpu_num': 0,
    'eval_path': '/home/euiseokjeong/Desktop/IMLAB/2022_ABAW_imlab/NAS/2022/result/2022_3_18_15_32_5',

    # train
    'epochs':20,
    'batch_size':256,
    'early_stop': 5,
    'generation': 1,
    'learning_rate': 0.0001,

    # model
    'feature_extractor_layers':[1024,512],
    'classifier_layers': [256,128],
    'domain_layers': [256, 128],
    'lstm_num': 512,
    'dropout_rate':0.2,

    # hyper parametre
    'temperature':1.5,
    'alpha':1.5,
    'beta':1,
    'gamma':1,
    'au_threshold':0.5,

    # task_weight
    'task_weight_exp':'as',
    'task_weight_flag':False,

     # domain adaptation
    'domain_weight':1,
    'adaptation_factor':1,
    'exp_domain_weight':True
}
