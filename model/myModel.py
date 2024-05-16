from model.net import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils.parameters as params
from utils.myUtils import MyUtils

class MyModel(object):
    FNAME_SAVED_MODEL = params.FNAME_SAVED_MODEL
    FNAME_LOAD_MODEL = params.FNAME_LOAD_MODEL
    FNAME_PREDICTION = params.FNAME_PREDICTION
    MODAL_NAME = params.NET
    EPOCHS = params.N_EPOCH

    def __init__(self):
        # defining the model
        self.model = globals()[self.MODAL_NAME]()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        trainable_params = []
        # freeze resnet parameters
        for n, p in self.model.named_parameters():
            if 'resnet' not in n:
                trainable_params.append(p)
        # defining the optimizer
        self.optimizer = optim.Adam(trainable_params, lr = 0.001)
        # defining the loss function
        self.criterion = nn.MSELoss(reduction='none')

    def fit(self, train_loader, valid_loader, epochs = EPOCHS, VALIDATION = True, SAVE = True, ANCHOR_TRAIN_LOSS = False, LOAD_MODEL = False):
        # checking if GPU is available
        if LOAD_MODEL:
            self._load_checked_point(self.FNAME_LOAD_MODEL)
        print(self.model.__class__)
        
        min_loss = np.inf

        valid_loss_arr, train_loss_arr = [], []
        for e in range(epochs):
            train_loss = 0.0
            self.model.train()     # Optional when not using Model Specific layer
            for data, label in train_loader:
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                self.optimizer.zero_grad()
                target = self.model(data)
                
                # print (self.model.cnn_layers[0].weight.max())
                
                loss = self.criterion(target, label)

                # loss = self.weightedLoss(loss)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.detach()
            train_loss /= len(train_loader)
            train_loss_arr.append(train_loss)
            if VALIDATION:
                valid_loss, _ = self._eval(valid_loader)
                print(f'Epoch {e+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
                valid_loss_arr.append(valid_loss)

                if ANCHOR_TRAIN_LOSS:
                    anchor_loss = train_loss
                else:
                    anchor_loss = valid_loss
                
                if SAVE and min_loss > anchor_loss:
                    self._save_checked_point(min_loss, anchor_loss)
                    min_loss = anchor_loss
            else:
                print(f'Epoch {e+1} \t\t Training Loss: {train_loss}')
                if SAVE and min_loss > train_loss:
                    self._save_checked_point(min_loss, train_loss)
                    min_loss = train_loss

        msg = ""
        if len(valid_loss_arr) > 0:
            msg += "validation loss: min = {}, average = {}".format(min(valid_loss_arr), sum(valid_loss_arr)/len(valid_loss_arr))
        if len(train_loss_arr) > 0:
            msg += " training loss: min = {}, average = {}".format(min(train_loss_arr), sum(train_loss_arr)/len(train_loss_arr))
        print(msg)
        return msg

    def weightedLoss(self, loss, scaler = 3):
        alpha = np.ones(63)
        for i in [6, 7, 8, 18, 19, 20, 30, 31, 32, 42, 43, 44, 57, 58, 59]:#5tips x (x, y, z)
            alpha[i] = scaler
        alpha = torch.tensor(alpha).to(target.device).unsqueeze(0).float()
        loss = loss * alpha
        return loss

    def _save_checked_point(self, loss_old, loss_new):
        print(f'Loss Decreased({loss_old:.6f}--->{loss_new:.6f}) \t Saving The Model in {self.FNAME_SAVED_MODEL}')
        # Saving State Dict
        torch.save(self.model.state_dict(), self.FNAME_SAVED_MODEL)
    
    def _eval(self, dataloader):
        #validation
        valid_loss = 0.0
        self.model.eval()# Optional when not using Model Specific layer

        y_predict, n_sample = [], 0
        for data, label in dataloader:
            n_sample += len(data) 
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                target = self.model(data)
                loss = self.criterion(target, label)
                valid_loss += loss.sum().item() #mean later over all loss can reduce error in float-point
                for i in target.data.cpu().numpy(): y_predict.append(i)
        valid_loss = valid_loss/n_sample/loss.shape[1]#loss.shape = (batch_size, y.shape)

        return valid_loss, np.array(y_predict)
    
    def _load_checked_point(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location = torch.device('cpu')))
        print("loaded {}".format(filename))

    def predict(self, dataloader, filename = FNAME_LOAD_MODEL, SAVE = False):
        # Load saved model
        self._load_checked_point(filename)
        
        predict_loss, y_predict = self._eval(dataloader)
        print("prediction loss = {}, y_predit = {}".format(predict_loss, y_predict.shape))
        if SAVE:
            fname = self.FNAME_PREDICTION
#             MyUtils.dumpPklWithTimestamp(y_predict, fname)
            MyUtils.dumpPkl(y_predict, fname)

        return y_predict, predict_loss
