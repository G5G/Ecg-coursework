import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from skimage.filters import threshold_multiotsu
#1st column = 1 = healthy, 2,3,4,5 = unhealthy
#2nd-all to the end = timesteps
#each row is recording session

#healthy training 2101, validation 234 
#test healthy 584, unhealthy 2081 (total 2665)
#total healthy data = 2919, unhealthy data = 2081
datafolder = 'data/ECG5000/'
#datafolder = 'data/TwoLeadECG/'
trainset_path = os.path.join(datafolder, 'ECG5000_TRAIN.txt')
testset_path = os.path.join(datafolder, 'ECG5000_TEST.txt')
#trainset_path = os.path.join(datafolder, 'TwoLeadECG_TRAIN.txt')
#testset_path = os.path.join(datafolder, 'TwoLeadECG_TEST.txt')
train_data = pd.read_csv(trainset_path, header=None, delim_whitespace=True)
test_data = pd.read_csv(testset_path, header=None, delim_whitespace=True)
combined_data = pd.concat([train_data, test_data], axis=0).values

healthy_data = combined_data[combined_data[:, 0] == 1]
unhealthy_data = combined_data[combined_data[:, 0] != 1]
class ECGDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data[:, 1:], dtype=torch.float32)
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min()) 
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

healthy_dataset = ECGDataset(healthy_data)
unhealthy_dataset = ECGDataset(unhealthy_data)

num_healthy_train = 2101
num_healthy_val = 234
num_healthy_test = 584

#num_healthy_train = 23
#num_healthy_val = 10
#num_healthy_test = 548#total 581 healthy and 581 unhealthy

train_healthy, val_healthy, test_healthy = random_split(
    healthy_dataset, [num_healthy_train, num_healthy_val, num_healthy_test])
test_dataset = ConcatDataset([test_healthy, unhealthy_dataset])
batch_size = 1

train_loader = DataLoader(train_healthy, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_healthy, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Testing 2nd dataset
dataset_folder = 'data/TwoLeadECG/'
testPath = os.path.join(dataset_folder, 'TwoLeadECG_TEST.txt')
test_data2 = pd.read_csv(testPath, header=None, delim_whitespace=True)
test_data2 = ECGDataset(test_data2.values)
test_loader2 = DataLoader(test_data2, batch_size=batch_size, shuffle=True)
print("DataLoader instances created")
    

class ecgnet(nn.Module):
    def __init__(self):
        super(ecgnet, self).__init__()

        #encoder
        self.lstm1 = nn.LSTM(1, 128,batch_first=True)
        self.lstm2 = nn.LSTM(128, 64,batch_first=True)
        self.lstm3 = nn.LSTM(64, 32,batch_first=True)
        
        self.fc1 = nn.Linear(32, 32)
        #decoder
        self.lstm4 = nn.LSTM(32, 64,batch_first=True)
        self.lstm5 = nn.LSTM(64, 128,batch_first=True)
        self.lstm6 = nn.LSTM(128, 1,batch_first=True)
        
        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        #encoder 
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        
        x, (hn,cn) = self.lstm3(x)
        hn = self.fc1(hn[-1])
        hn = hn.unsqueeze(1).repeat(1, x.size(1), 1)
        x = hn
        #decoder
        x, _ = self.lstm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x, _ = self.lstm5(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x, _ = self.lstm6(x)
        return x
    
    
epochs = 200
model = ecgnet()
device = torch.device('cuda')
model = model.to(device)

criterion = nn.L1Loss()#sum

optimizer = optim.Adam(model.parameters(), lr=0.001)
outp = []
inp = []
train_losses = []
val_losses =[]
def train(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data,_ in train_loader:
            
            data = data.to(device)
            data = data.unsqueeze(-1).to(device)
            output = model(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            outp.append(output)
            inp.append(data)
            #print(loss.item())
        train_loss /= len(train_loader)   
        train_losses.append(train_loss)             
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for dataa,_ in val_loader:
                
                dataa = dataa.to(device)
                dataa = dataa.unsqueeze(-1).to(device)
                
                output = model(dataa)
                loss = criterion(output, dataa)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if epoch % 199 == 0:#using this for debugging -----------------------
            with torch.no_grad():
                model.eval()
                data = test_loader.dataset[100][0].unsqueeze(0).unsqueeze(-1).to(device)
                
                
                output = model(data)
                data1 = test_loader.dataset[10][0].unsqueeze(0).unsqueeze(-1).to(device)
                output1 = model(data1)
                data2 = val_loader.dataset[5][0].unsqueeze(0).unsqueeze(-1).to(device)
                output2 = model(data2)
                
                plot1 = plt.subplot2grid((3, 3), (0, 0))
                plot1.set_title('Healthy Test Data Epoch: ' + str(epoch))
                plot2 = plt.subplot2grid((3, 3), (0, 2))
                plot2.set_title('Unhealthy Test Data Epoch: ' + str(epoch))
                plot3 = plt.subplot2grid((3, 3), (1, 0))
                plot3.set_title('Healthy Val Data Epoch: ' + str(epoch))
                
                plot1.plot(output.cpu().detach().numpy()[0, :, 0], color='red', label='output')
                plot1.plot(data.cpu().detach().numpy()[0, :, 0], color='green', label='input')
                plot1.legend()

                plot2.plot(output1.cpu().detach().numpy()[0, :, 0], color='red', label='output')
                plot2.plot(data1.cpu().detach().numpy()[0, :, 0], color='green', label='input')
                plot2.legend()

                plot3.plot(output2.cpu().detach().numpy()[0, :, 0], color='red', label='output')
                plot3.plot(data2.cpu().detach().numpy()[0, :, 0], color='green', label='input')
                plot3.legend()
                
                
                plt.tight_layout()
                plt.show()
                #----------------------------------------------------------------
        print("Epoch: " +  str(epoch) + " Training loss: " + str(train_loss) + " Validation loss: " + str(val_loss))
    torch.save(model.state_dict(), 'model.pth')
    torch.save(model, 'model.pth')
    #plot train_losses and val_losses on same graph
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
      
train(model, train_loader, val_loader, epochs, criterion, optimizer, device)


def autothreshold(losses):
    thresholds = threshold_multiotsu(np.array(losses), classes=2)
    threshol = thresholds[0]
    return threshol

def visualize_histogram_with_threshold(losses, num_bins=256):
    hist, bin_edges = np.histogram(losses, bins=num_bins, range=(min(losses), max(losses)), density=True)
    threshold = autothreshold(losses)
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=num_bins, range=(min(losses), max(losses)), density=True, alpha=0.75, color='blue', label='Losses Histogram')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')

    plt.xlabel('Losses')
    plt.ylabel('Density')
    plt.title('Histogram of Losses with Optimal Threshold')
    plt.legend()
    plt.show()

def test(model, testloader):
    model.eval()
    with torch.no_grad():
        labels = []
        predictions = []
        losses = []
        for data, label in testloader:
            data = data.unsqueeze(-1).to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, data)
            losses.append(loss.item())
            labels.append(1 if label == 1 else 0)
        visualize_histogram_with_threshold(losses)
        optimal_threshold = autothreshold(losses)
        print(f"Optimal Threshold: {optimal_threshold}")
        for loss in losses:
            if loss < optimal_threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        confusion = confusion_matrix(labels, predictions)
        plt.matshow(confusion)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion[i, j], ha='center', va='center', color='red')
        plt.show()
        print(confusion)
        Precision, Recall, F1sc, sup = precision_recall_fscore_support(labels, predictions, average='binary')
        print("Precision: " + str(Precision), " Recall: " + str(Recall), " F1: " + str(F1sc))
        accuracy = (confusion[0, 0] + confusion[1, 1]) / np.sum(confusion)
        print("Accuracy: " + str(accuracy))
        
    
model = torch.load('model.pth')
test(model, test_loader)




        
