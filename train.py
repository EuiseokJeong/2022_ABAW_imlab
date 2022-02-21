from trainer import Trainer
from dataloader import dataloader
from config import configs
from models import get_model

def main():
    batch_size = configs['batch_size']
    epochs = configs['epochs']
    # train_dataloader = dataloader(type='train', batch_size=batch_size)
    # valid_dataloader = dataloader(type='valid', batch_size=batch_size)
    teacher = get_model(configs)
    student = get_model(configs)

    trainer = Trainer(teacher, student, alpha=1.5, beta=0.5, gamma=1, batch_size=batch_size)
    trainer.train_teacher(epochs)
    trainer.train_student(epochs)



if __name__ == '__main__':
    main()


