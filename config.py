configs = {
    'stride':10,
    'time_window':2,
# /home/euiseokjeong/Desktop/IMLAB/ABAW/data/2022
    'data_path': '/home/euiseokjeong/Desktop/IMLAB/ABAW/data/2022',
    'limit_gpu':1024*2,
    'epochs':50,
    'batch_size':256,
    'feature_extractor_layers':[1024,512],
    'classifier_layers': [256,128],
    'lstm_num': 512,
    'dropout_rate':0.2,
    'temperature':1.5,
    'early_stop':5,
    'generation':5,
    'alpha':1.5,
    'beta':0.5,
    'gamma':1,
    'au_threshold':0.5,
    'learning_rate':0.00001,
    'eval_path':'/home/euiseokjeong/Desktop/IMLAB/ABAW/result/2022_3_5_21_44_27'
}
