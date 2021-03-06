# Multitask Emotion Recognition Model with Knowledge Distillation and Task Discriminator
Source code of [Multitask Emotion Recognition Model with Knowledge Distillation and Task Discriminator](https://arxiv.org/abs/2203.13072)

### Requirements

1. requirements
    
    - Anaconda must be installed and GPU-enabled.
    - conda_requirements.yml should be fixed to fit your anaconda path.
    
    ```python
    name: imlab_ABAW
    channels:
      - conda-forge
        .
        .
        .
    # specify your anaconda path with env name
    # ex)/home/{user_name}/anaconda3/envs/imlab_ABAW
    prefix: /home/{your_user_name}/anaconda3/envs/imlab_ABAW
    ```
    
    - Then, you can create new conda environment.
    
    ```python
    $ conda env create -f conda_requirements.yml
    $ conda activate imlab_ABAW
    $ pip install -r requirements.txt
    ```
    

## Download data

1. Download data in ./data directory(data/)
    - Download extracted feature from SoundNet, FER model. You can download here ([image_feature](https://www.dropbox.com/s/eaq76d5xouo5glu/image_t%282%29_s%2810%29.zip?dl=0), [audio_feature](https://www.dropbox.com/s/zzcll6sk04jva3x/audio.zip?dl=0))
    - You should download idx pickle including file_name, labels etc. You can download [here](https://www.dropbox.com/s/s6cz3f1ivce5xai/idx.zip?dl=0)
    - You should also download Multi_Task_Learning_Challenge_test_set_release.txt. download [here](https://www.dropbox.com/s/cvfyq6knsr9bkzy/testset.zip?dl=0)
    
    ```markdown
    data_path
    	-features
    	-idx
    	-testset
    	-result(will be generated automatically)
    ```
    
2. Change ‘data_path’ in config.py to your data_path data downloaded. 

## Train

1. When you run trainer.py, it is stored in a directory called result in the data_path as the date and time of the file execution.
    - If you want a pre-trained file,  you can download [here](https://www.dropbox.com/s/sb5trorb50kwacq/pre_trained.zip?dl=0)

## Evaluate

1. Change "eval_path" on config.py to the absolute path of generated folder.
2. Running submit.py creates a submission folder in that path and creates a submission file.

## Citation
@misc{https://doi.org/10.48550/arxiv.2203.13072,
  doi = {10.48550/ARXIV.2203.13072},
  url = {https://arxiv.org/abs/2203.13072},
  author = {Jeong, Euiseok and Oh, Geesung and Lim, Sejoon},
  title = {Multitask Emotion Recognition Model with Knowledge Distillation and Task Discriminator},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Zero v1.0 Universal}
}
