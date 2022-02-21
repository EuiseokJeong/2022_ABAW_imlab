from trainer import Trainer
from dataloader import dataloader
from config import configs
from models import get_model
from utils import get_result_path, check_and_limit_gpu

def main():
    batch_size = configs['batch_size']
    epochs = configs['epochs']
    data_path = configs['data_path']
    limit_gpu = configs['limit_gpu']
    result_path, weight_path, src_path = get_result_path(data_path)

    check_and_limit_gpu(limit_gpu)
    teacher = get_model(configs)
    student = get_model(configs)

    trainer = Trainer(teacher, student, alpha=1.5, beta=0.5, gamma=1, batch_size=batch_size, gen=1)
    trainer.train_teacher(epochs)
    trainer.train_student(epochs)



if __name__ == '__main__':
    main()


