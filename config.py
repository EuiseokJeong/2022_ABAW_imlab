configs = {
    'stride':10,
    'time_window':2,
# /home/euiseokjeong/Desktop/IMLAB/ABAW/data/2022
    'data_path': '/home/euiseokjeong/Desktop/IMLAB/ABAW/data/2022',
    'limit_gpu':1024*3,
    'epochs':20,
    'batch_size':256,
    'feature_extractor_layers':[1024,512],
    'classifier_layers': [256,128],
    'lstm_num': 512,
    'dropout_rate':0.5,
    'temperature':3,
    'early_stop':5,
    'generation':1,
    'alpha':1.5,
    'beta':0.5,
    'gamma':1,
    'au_threshold':0.5,
    'learning_rate':0.0001,
    'eval_path':'/home/euiseokjeong/Desktop/IMLAB/ABAW/result/keep/temperature/2022_3_5_21_44_27(t_1-5)'
}
