import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):

        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight
        self.size_average=size_average
    def forward(self, output, y):
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)
        loss=output-output_y+self.margin
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        loss[loss<0]=0
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        if(self.weight is not None):
            loss=loss*self.weight
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]
        return loss

class Trainer(object):
    """Trainer class to abstract rudimentary training loop."""

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            device):
        """Set trainer class with model, criterion, optimizer. (Data is passed to train/eval)."""
        super(Trainer, self).__init__()
        self.model= model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lda = 1e-20 # lda = 1e-18 
    def train(self, loader,type):
        """Train model using batches from loader and return accuracy and loss."""
        total_loss, total_acc = 0.0, 0.0
        self.model.train()
        for _, (inputs, targets) in tqdm(enumerate(loader), total=len(loader), desc='Training'):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            if type == "softmax":
                loss = self.criterion(outputs, targets)
            elif type == "cxsvm1":
                regularization_loss = (1/2)*torch.sum(torch.square(self.model.module.fc.weight))
                loss = self.lda*regularization_loss + multiClassHingeLoss()(outputs, targets)
            elif type == "cxsvm2":
                regularization_loss = (1/2)*torch.max(torch.sum(torch.square(self.model.module.fc.weight),1))
                loss = self.lda*regularization_loss + multiClassHingeLoss()(outputs, targets)
            elif type == "cxsvm3":
                regularization_loss = (1/2)*torch.sum(torch.abs(self.model.module.fc.weight))
                loss = self.lda*regularization_loss + multiClassHingeLoss()(outputs, targets)
            elif type == "cxsvm4":
                regularization_loss = (1/2)*torch.max(torch.sum(torch.abs(self.model.module.fc.weight),1))
                loss = self.lda*regularization_loss + multiClassHingeLoss()(outputs, targets)     
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total_loss += loss.item()
            total_acc += (predicted == targets).float().sum().item() / targets.numel()
        return total_loss / len(loader), 100.0 * total_acc / len(loader)

    def test(self, loader, type):
        """Evaluate model using batches from loader and return accuracy and loss."""
        with torch.no_grad():
            total_loss, total_acc = 0.0, 0.0
            self.model.eval()
            for _, (inputs, targets) in tqdm(enumerate(loader), total=len(loader), desc='Testing '):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                if type == "softmax":
                    loss = self.criterion(outputs, targets)
                    outputs = F.softmax(outputs,dim=1)
                elif type == "cxsvm1":
                    regularization_loss = (1/2)*torch.sum(torch.square(self.model.module.fc.weight))
                    loss = self.lda*regularization_loss + multiClassHingeLoss()(outputs, targets)
                elif type == "cxsvm2":
                    regularization_loss = (1/2)*torch.max(torch.sum(torch.square(self.model.module.fc.weight),1))
                    loss = self.lda*regularization_loss + multiClassHingeLoss()(outputs, targets)
                elif type == "cxsvm3":
                    regularization_loss = (1/2)*torch.sum(torch.abs(self.model.module.fc.weight))
                    loss = self.lda*regularization_loss + multiClassHingeLoss()(outputs, targets)
                elif type == "cxsvm4":
                    regularization_loss = (1/2)*torch.max(torch.sum(torch.abs(self.model.module.fc.weight),1))
                    loss = self.lda*regularization_loss + multiClassHingeLoss()(outputs, targets) 
                _, predicted = torch.max(outputs, 1)
                total_loss += loss.item()
                total_acc += (predicted == targets).float().sum().item() / targets.numel()
        return total_loss / len(loader), 100.0 * total_acc / len(loader)
