import os

import torch

from torch.nn import DataParallel
from torch.utils.data import random_split

from module.eval import evalDic
from module.models import modelDic

from utils import UrbanSound, ESC50Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trainer:
    def __init__(self, args):
        for key in args:
            setattr(self, key, args[key])
        
        self.model_path = 'models/'+self.dataset+'/'+self.model_name+'/'+self.run_name+'.pt'
        if not os.path.exists('models/'+self.dataset+'/'+self.model_name+'/'):
            os.makedirs('models/'+self.dataset+'/'+self.model_name+'/')
        print(f'[DEVICE][{DEVICE}]')
        print(f'[DEVICE Count][{torch.cuda.device_count()}]')
        print(f'[Model][Saved at {self.model_path}]')

    def _setDataset(self):
        if self.dataset == 'urbansound':
            traindataset = UrbanSound([1,2,3,4,5,6,7,8,9,10], self.targetLength)

        elif self.dataset == 'esc50':
            traindataset = ESC50Dataset(self.targetLength)

        self.classes = traindataset.numclasses
            
        train_len = int(0.8 * len(traindataset))
        val_len = (len(traindataset) - train_len)//2
        test_len = len(traindataset) - train_len - val_len

        traindataset, validdataset, testdataset = random_split(traindataset, [train_len, val_len, test_len])
        
        self.trainloader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, num_workers = self.workers, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(validdataset, batch_size=self.batch_size, num_workers = self.workers, shuffle=False)
        self.testloader = torch.utils.data.DataLoader(testdataset, batch_size=self.batch_size, num_workers = self.workers, shuffle=False)

        if self.model_name != 'astmodel':
            self.memloader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, num_workers = self.workers, shuffle=True)

    def _setModelAndOptimizer(self):
        self.model = modelDic[self.model_name](self.classes).to(DEVICE)
        self.model = DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _setLossFunction(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _train(self):
        accuracy = evalDic['accuracy']
        bestAcc = 0
        for epoch in range(self.epochs):
            self.model.train()
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

            trainAcc = accuracy(self.trainloader, self.model, DEVICE)
            validAcc = accuracy(self.validloader, self.model, DEVICE)
            print(f'Epoch: {epoch+1}, Training Accuracy: {trainAcc} Validation Accuracy: {validAcc}')
            if validAcc > bestAcc:
                bestAcc = validAcc
                torch.save(self.model.state_dict(), self.model_path)

    def _test(self, path=None):
        accuracy = evalDic['accuracy']
        if path == None:
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(path))
        testAcc = accuracy(self.testloader, self.model, DEVICE)
        print(f'Test Accuracy: {testAcc}')

    def _trainCustom(self):
        accuracy = evalDic['accuracycustom']
        bestAcc = 0
        for epoch in range(self.epochs):
            self.model.train()
            for data, mem in zip(self.trainloader, self.memloader):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                mem, _ = mem
                mem = mem.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(inputs, mem)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

            trainAcc = accuracy(self.trainloader, self.memloader, self.model, DEVICE)
            validAcc = accuracy(self.validloader, self.memloader, self.model, DEVICE)
            print(f'Epoch: {epoch+1}, Training Accuracy: {trainAcc} Validation Accuracy: {validAcc}')
            if validAcc > bestAcc:
                bestAcc = validAcc
                torch.save(self.model.state_dict(), self.model_path)
        
    def _testCustom(self, path=None):
        accuracy = evalDic['accuracycustom']
        if path == None:
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(path))
        testAcc = accuracy(self.testloader, self.memloader, self.model, DEVICE)
        print(f'Test Accuracy: {testAcc}')

    def run(self):
        self._setDataset()
        print(f'[Dataset][{self.dataset}]')
        self._setModelAndOptimizer()
        print(f'[Model][{self.model_name}]')
        self._setLossFunction()
        if self.model_name == 'astmodel':
            self._train()
            self._test()
        else:
            self._trainCustom()
            self._testCustom()
    
    def eval(self):
        self._setDataset()
        self._setModelAndOptimizer()
        if self.model_name == 'astmodel':
            self._test(self.pretrained_path)
        else:
            self._testCustom(self.pretrained_path)
