import logging
import random
from typing import Any, Dict
from PIL import Image
from torch.utils.data import Dataset
import os
from bcnn import BilinearModel, Trainer
from config import parse_args_main
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.StreamHandler(),
    ])
logger = logging.getLogger()

class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, training=False):
        self.image_list = []
        self.id_list = []
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 0
        self.training = training
        with open(txt_file, 'r') as f:
            line = f.readline()
            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])
                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()
            f.close()
        self.num_classes = max(self.id_list)+1
        
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.id_list[idx]
        img_name = os.path.join(self.root_dir,img_name)
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image,label
def checkpoint(
        trainer,
        epoch,
        accuracy,
        savedir,
        config):
    """Save a model checkpoint at specified location."""
    checkpoint: Dict[str, Any] = {
        "model": trainer.model.state_dict(),
        "optim": trainer.optimizer.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "config": config,
    }
    logger.debug("==> Checkpointing Model")
    torch.save(checkpoint, savedir / 'checkpoint_test.pt')


def run_epochs_for_loop(
        trainer,
        epochs,
        train_loader,
        test_loader,
        savedir,
        config,
        scheduler,
        type_classifier):
    """Run train + evaluation loop for specified epochs.

    Save checkpoint to specified save folder when better optimum is found.
    If LR scheduler is specified, change LR accordingly.
    """
    best_acc = 0.0
    train_loss_all, train_acc_all = [], []
    test_loss_all, test_acc_all = [], []
    for epoch in range(epochs):
        (train_loss, train_acc) = trainer.train(train_loader,type_classifier)  
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        (test_loss, test_acc) = trainer.test(test_loader,type_classifier)  
        test_loss_all.append(test_loss)
        test_acc_all.append(test_acc)
        logger.info("Epoch %d: TrainLoss %f \t TrainAcc %f" % (epoch, train_loss, train_acc))
        logger.info("Epoch %d: TestLoss %f \t TestAcc %f" % (epoch, test_loss, test_acc))
        if scheduler is not None:
            scheduler.step(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint(trainer, epoch, test_acc, savedir, config)
    t = [i for i in range(epochs)]
    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 20})
    plt.plot(t,train_acc_all, label = "Accuracy_train",linewidth=2,ls='-.')
    plt.plot(t,test_acc_all, label = "Accuracy_test",linewidth=2,ls='-.')
    plt.rcParams.update({'font.size': 20})
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("ACC_sq_svm.png")

    t = [i for i in range(epochs)]
    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 20})
    plt.plot(t,train_loss_all, label = "Loss_train",linewidth=2,ls='-.')
    plt.plot(t,test_loss_all, label = "Loss_test",linewidth=2,ls='-.')
    plt.rcParams.update({'font.size': 20})
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("Loss_sq_svm.png")


def main():
    """Train bilinear CNN."""
    args = parse_args_main()
    logger.debug(args)

    # random seeding
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if len(args.gpus) > 0:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        #device = torch.device(args.device)
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("GPU name: ",torch.cuda.get_device_name(0))
    args.savedir.mkdir(parents=True, exist_ok=True)
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    img_size = 448
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.Normalize(rgb_mean, rgb_std),
        transforms.RandomCrop(img_size),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
        transforms.CenterCrop(img_size),
    ])
    root = os.path.join(args.datadir,"images")
    txt_train = os.path.join(args.datadir,"aircraft_train.txt")
    txt_test = os.path.join(args.datadir,"aircraft_test.txt")
    type_classifier = args.type_cls
    train_set = CustomDataset(txt_train,root,transform_train,True)
    test_set = CustomDataset(txt_test,root,transform_test,False)
    train_loader = DataLoader(train_set,batch_size=args.batchsize,num_workers=args.workers,shuffle=True,pin_memory=True,)
    test_loader = DataLoader(test_set,batch_size=args.batchsize,num_workers=args.workers,shuffle=False,pin_memory=True,)


    criterion = nn.CrossEntropyLoss()
    model = BilinearModel(num_classes=100)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    criterion = criterion.to(device)

    logger.debug("==> PRETRAINING NEW BILINEAR LAYER ONLY")
    for param in model.module.features.parameters():
        param.requires_grad = False
    optimizer = optim.SGD(
        model.module.fc.parameters(),
        lr=args.lr[0],
        weight_decay=args.wd[0],
        momentum=args.momentum,
        nesterov=True,
    )
    pretrainer = Trainer(
        model,
        criterion,
        optimizer,
        device,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.stepfactor,
        patience=args.patience,
        verbose=True,
        threshold=1e-4,
    )
    run_epochs_for_loop(
        trainer=pretrainer,
        epochs=args.epochs[0],
        train_loader=train_loader,
        test_loader=test_loader,
        savedir=args.savedir,
        config=args,
        scheduler=scheduler,
        type_classifier = type_classifier,
    )
    # logger.debug("==> FINE-TUNING OLDER LAYERS AS WELL")
    # for param in model.module.features.parameters():
    #     param.requires_grad = True
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=args.lr[1],
    #     weight_decay=args.wd[1],
    #     momentum=args.momentum,
    #     nesterov=True,
    # )
    # finetuner = Trainer(
    #     model,
    #     criterion,
    #     optimizer,
    #     device,
    # )
    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=args.stepfactor,
    #     patience=args.patience,
    #     verbose=True,
    #     threshold=1e-4,
    # )
    # run_epochs_for_loop(
    #     trainer=finetuner,
    #     epochs=args.epochs[1],
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     savedir=args.savedir,
    #     config=args,
    #     scheduler=scheduler,
    # )


if __name__ == "__main__":
    main()
