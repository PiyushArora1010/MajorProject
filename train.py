import os

import torch

from torch.utils.data import random_split

from module.eval import accuracy
from module.models import modelDic

from utils import UrbanSound, ESC50Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trainer:
    def __init__(self, args):
        for key in args:
            setattr(self, key, args[key])
        
        if not os.path.exists('models'):
            os.makedirs('models')


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

    def _setModelAndOptimizer(self):
        self.model = modelDic[self.model_name](self.classes).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _setLossFunction(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _train(self):
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
                torch.save(self.model.state_dict(), 'models/'+self.run_name+'.pt')

    def _test(self):
        self.model.load_state_dict(torch.load(self.model_path))
        testAcc = accuracy(self.testloader, self.model, DEVICE)
        print(f'Test Accuracy: {testAcc}')

    def run(self):
        self._setDataset()
        self._setModelAndOptimizer()
        self._setLossFunction()
        self._train()
        self._test()