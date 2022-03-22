configs = {
    # setting
    'stride':10,
    'time_window':2,
    'data_path': '/home/euiseokjeong/Desktop/IMLAB/2022_ABAW_imlab/data/2022',

    'limit_gpu':1024*1,
    'gpu_num': 0,
    'eval_path': '/home/euiseokjeong/Desktop/IMLAB/2022_ABAW_imlab/NAS/2022/result/2022_3_20_2_30_59',


    # train
    'epochs':20,
    'batch_size':256,
    'early_stop': 5,
    'generation': 5,
    'learning_rate': 0.0001,

    # model
    'feature_extractor_layers':[1024,1024,1024,512,512,512],
    'classifier_layers': [256,128],
    'domain_layers': [256, 128],
    'lstm_num': 512,
    'dropout_rate':0.5,

    # hyper parametre
    'temperature':2.5,
    'alpha':10,
    'beta':0.9,
    'gamma':1,
    'au_threshold':0.5,

    # task_weight
    'task_weight_exp':0.5,
    'task_weight_flag':True,

     # domain adaptation


    'domain_weight':0,
    'adaptation_factor':0,
    'exp_domain_weight':False

}
