configs = {
    'stride':10,
    'time_window':2,
# /home/euiseokjeong/Desktop/IMLAB/ABAW/data/2022
    'data_path': '/home/euiseokjeong/Desktop/IMLAB/ABAW/data/2022',
    'limit_gpu':1024*2,
    'epochs':30,
    'batch_size':128,
    'feature_extractor_layers':[512, 256],
    'classifier_layers': [128],
    'latm_layers': [256],
    'dropout_rate':0.2,
    'temperature':1.5,
    'early_stop':5,
    'generation':5,
    'alpha':1.5,
    'beta':0.5,
    'gamma':1
}