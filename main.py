import os
import torch 
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from skimage.filters import threshold_multiotsu
from sklearn.model_selection import train_test_split
import random
#1st column = 1 = healthy, 2,3,4,5 = unhealthy
#2nd-all to the end = timesteps
#each row is recording session

#healthy training 2101, validation 234 
#test healthy 584, unhealthy 2081 (total 2665)
#total healthy data = 2919, total unhealthy data = 2081


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
usr_inp = input("Disable randomness? (y/n): ")
if usr_inp == 'y':
    set_seed()
    print("Randomness disabled for debugging")
else:
    print("Randomness enabled")



datafolder = 'data/ECG5000/'
trainset_path = os.path.join(datafolder, 'ECG5000_TRAIN.txt')
testset_path = os.path.join(datafolder, 'ECG5000_TEST.txt')
train_data = pd.read_csv(trainset_path, header=None, delim_whitespace=True)
test_data = pd.read_csv(testset_path, header=None, delim_whitespace=True)
combined_data = pd.concat([train_data, test_data], axis=0).values

healthy_data = combined_data[combined_data[:, 0] == 1]#shape = (2919, 141)
unhealthy_data = combined_data[combined_data[:, 0] != 1]#shape = (2081, 141)

x = unhealthy_data[:, 1:]
y = unhealthy_data[:, 0]
X_main, x_x, y_main, y_y = train_test_split(x, y, test_size=0.3, stratify=y)
X_train, X_testx, y_train, y_testx = train_test_split(x_x, y_y, test_size=0.5, stratify=y_y)
X_test, X_val, y_test, y_val = train_test_split(X_testx, y_testx, test_size=0.5, stratify=y_testx)
train_unh = np.column_stack((y_train, X_train))
val_unh = np.column_stack((y_val, X_val))
test_unh = np.column_stack((y_test, X_test))
unhealthy_data = np.column_stack((y_main, X_main))

train_unh_labels = train_unh[:, 0]
train_unh_data = train_unh[:, 1:]
R_T_ind = np.where(train_unh_labels == 2)[0]  # Indices of R-T samples
other_ind = np.where(train_unh_labels != 2)[0]  # Indices of all other classes
R_T_keep = 40  # Number of R-T samples to keep
R_T_ind_sub = np.random.choice(R_T_ind, R_T_keep, replace=False)  # Randomly select R-T samples
tmp_ind = np.concatenate((R_T_ind_sub, other_ind))
reduced_train_unh_data = train_unh_data[tmp_ind]
reduced_train_unh_labels = train_unh_labels[tmp_ind]
train_unh = np.column_stack((reduced_train_unh_labels, reduced_train_unh_data))





class ECGDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data[:, 1:], dtype=torch.float32)#removes the label 
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())#convert to 0-1 range 
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)#assings the label to the label variable

    def __len__(self):
        return len(self.data)#returns the length of the data

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]#returns the data and the label at the index
    
def add_noise(signal, noise_factor=0.05):
    noise = noise_factor * np.random.randn(*signal.shape)#adds noise to the signal
    return signal + noise

def time_shift(signal, shift_max=5):
    shift = np.random.randint(-shift_max, shift_max)#randomly shifts the signal
    return np.roll(signal, shift)

def scale(signal, scale_factor=0.1):
    scaling_factor = np.random.uniform(1 - scale_factor, 1 + scale_factor)#randomly scales the signal
    return signal * scaling_factor
#--------------------------------------------------------
def generate_augmented_data(dataset, num_copies=20):
    augmented_data = []
    augmented_labels = []

    for i in range(num_copies):
        for x in range(len(dataset)):
            signal, label = dataset[x]  
            signal2 = signal.numpy()
            signal2= add_noise(signal2)
            signal2 = time_shift(signal2)
            signal2 = scale(signal2)

            augmented_data.append(signal2)
            augmented_labels.append(label.item())

    return np.array(augmented_data), np.array(augmented_labels)



healthy_dataset = ECGDataset(healthy_data)
unhealthy_dataset = ECGDataset(unhealthy_data)
train_unh = ECGDataset(train_unh)

augmented_signals, augmented_labels = generate_augmented_data(train_unh, num_copies=10)#generates augmented data
train_unh = np.column_stack((augmented_labels, augmented_signals))#combines the augmented data with the original data
train_unh = ECGDataset(train_unh)
val_unh = ECGDataset(val_unh)
test_unh = ECGDataset(test_unh)
a,b,c,d = (0,0,0,0)
for i in unhealthy_dataset.labels:
    if i == 2:
        a += 1
    elif i == 3:
        b += 1
    elif i == 4:
        c += 1
    elif i == 5:
        d += 1

print("Unhealthy_dataset:  R-T = ",a, " PVC = ",b," SB or EB = ",c," FVN = ",d)
a,b,c,d = (0,0,0,0)
for i in train_unh.labels:
    if i == 2:
        a += 1
    elif i == 3:
        b += 1
    elif i == 4:
        c += 1
    elif i == 5:
        d += 1

total = a+b+c+d
w0 = total/(a*4)#calculating the weight for each class due to class imbalance
w1 = total/(b*4)
w2 = total/(c*4)
w3 = total/(d*4)

print("unhealthy train:  R-T = ",a, " PVC = ",b," SB or EB = ",c," FVN = ",d)
a,b,c,d = (0,0,0,0)
for i in val_unh.labels:
    if i == 2:
        a += 1
    elif i == 3:
        b += 1
    elif i == 4:
        c += 1
    elif i == 5:
        d += 1


print("unhealthy val:  R-T = ",a, " PVC = ",b," SB or EB = ",c," FVN = ",d)
a,b,c,d = (0,0,0,0)
for i in test_unh.labels:
    if i == 2:
        a += 1
    elif i == 3:
        b += 1
    elif i == 4:
        c += 1
    elif i == 5:
        d += 1

print("unhealthy test:  R-T = ",a, " PVC = ",b," SB or EB = ",c," FVN = ",d)

num_healthy_train = 2101
num_healthy_val = 234
num_healthy_test = 584


#Unhealthy data meaning:
#R-on-T premature ventricular contraction 1767 samples [2]--identifier used within the dataset
#Premature Ventricular Contraction 96 samples [3]--identifier used within the dataset
#Supraventricular Premature or Ectopic beat 196 samples [4]--identifier used within the dataset
#Fusion of ventricular and Normal Beat 24 samples [5]--identifier used within the dataset
train_healthy, val_healthy, test_healthy = random_split(healthy_dataset, [num_healthy_train, num_healthy_val, num_healthy_test])
test_dataset = ConcatDataset([test_healthy, unhealthy_dataset])
batch_size = 1 #Leave it as 1 since ecg-net is using lstm and that's what they used in the paper

train_loader = DataLoader(train_healthy, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_healthy, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_unh = DataLoader(train_unh, batch_size=30, shuffle=True)
val_unh = DataLoader(val_unh, batch_size=30, shuffle=False)
test_unh = DataLoader(test_unh, batch_size=30, shuffle=False)

class ecgnet(nn.Module):
    def __init__(self):
        super(ecgnet, self).__init__()

        #encoder
        self.lstm1 = nn.LSTM(1, 128,batch_first=True)
        self.lstm2 = nn.LSTM(128, 64,batch_first=True)
        self.lstm3 = nn.LSTM(64, 32,batch_first=True)
        self.lstm33 = nn.LSTM(32,16,batch_first=True)
        #decoder
        self.lstm44 = nn.LSTM(16, 32,batch_first=True)
        self.lstm4 = nn.LSTM(32, 64,batch_first=True)
        self.lstm5 = nn.LSTM(64, 128,batch_first=True)
        self.lstm6 = nn.LSTM(128, 1,batch_first=True)

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
        
        x, _ = self.lstm3(x)
        x= self.relu(x)
        x= self.dropout(x)
        x,(hn,cn) = self.lstm33(x)

        
        #latent space 
        hn = hn.repeat(1, x.size(1), 1)#repeating the hidden state across all timesteps
        x = hn
        #decoder
        x, _= self.lstm44(x)
        x, _ = self.lstm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x, _ = self.lstm5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm6(x)
        
        return x
    
    
outp = []
inp = []
train_losses = []
val_losses =[]
def train(model, train_loader, val_loader, epochs, criterion, optimizer, device):#train ECG-NET
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
        
        if epoch % 99 == 0:#using this for debugging - plots the output of the model every x amount of epochs
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

        print("Epoch: " +  str(epoch) + " Training loss: " + str(train_loss) + " Validation loss: " + str(val_loss))
    torch.save(model.state_dict(), 'modell.pth')
    torch.save(model, 'model.pth')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
 
def weight_init(m):
    if isinstance(m,nn.Conv1d):#initializes the weights of the model
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        
    
class customnet(nn.Module):#ECG-TCN model
    def __init__(self):
        super(customnet, self).__init__()
        self.pad0 = nn.ConstantPad1d(padding = (10, 0), value = 0)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=11)#initial 1x1 convolution layer
        
        self.conv1_1 = nn.Conv1d(in_channels=2, out_channels=11, kernel_size=1)
        self.pad1 = nn.ConstantPad1d(padding = ((10) * 1, 0), value = 0)
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=11, kernel_size=11,dilation=1)
        self.pad2 = nn.ConstantPad1d(padding = ((10)*1,0), value = 0)
        self.conv2_2 = nn.Conv1d(in_channels=11, out_channels=11, kernel_size=11,dilation=1)
        
        self.pad3 = nn.ConstantPad1d(padding = ((10) * 2, 0), value = 0)
        self.conv3 = nn.Conv1d(in_channels=11, out_channels=11, kernel_size=11, dilation=2)
        self.pad4 = nn.ConstantPad1d(padding = ((10)*2,0), value = 0)
        self.conv3_2 = nn.Conv1d(in_channels=11, out_channels=11, kernel_size=11, dilation=2)
        
        self.pad5 = nn.ConstantPad1d(padding = ((10) * 4, 0), value = 0)
        self.conv4 = nn.Conv1d(in_channels=11, out_channels=11, kernel_size=11, dilation=4)
        self.pad6 = nn.ConstantPad1d(padding = ((10)*4,0), value = 0)
        self.conv4_2 = nn.Conv1d(in_channels=11, out_channels=11, kernel_size=11, dilation=4)
        self.batchnorm2 = nn.BatchNorm1d(11)
        self.batchnorm = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(11 * 140, 200, bias=False) 
        self.fc2 = nn.Linear(200, 40, bias=False)
        self.fc3 = nn.Linear(40, 4, bias=False)

    def forward(self, x):
        
        x = self.pad0(x)
        x = self.conv1(x) 
        x = self.batchnorm(x)
        x = self.relu(x)
        x_1 = x
        
        x_1 = self.relu(self.dropout(self.batchnorm2(self.conv2(self.pad1(x_1)))))
        x_1 = self.relu(self.dropout(self.batchnorm2(self.conv2_2(self.pad2(x_1)))))
        
        x = self.relu(self.batchnorm2(self.conv1_1(x)))

        x = x_1 + x
        
        x_1 = x
        x_1 = self.relu(self.dropout(self.batchnorm2(self.conv3(self.pad3(x_1)))))
        x_1 = self.relu(self.dropout(self.batchnorm2(self.conv3_2(self.pad4(x_1)))))
        
        x = x_1 + x
        
        x_1 = x
        x_1 = self.relu(self.dropout(self.batchnorm2(self.conv4(self.pad5(x_1)))))
        x_1 = self.relu(self.dropout(self.batchnorm2(self.conv4_2(self.pad6(x_1)))))
        
        x = x_1 + x 
        x = x.flatten(1)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
outp = []
inp = []
train_losses = []
val_losses =[]
train_accuracies = []
val_accuracies = [] 
def train2(model, train_loader, val_loader, epochs, criterion, optimizer, device):#train ECG-TCN

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy= 0
        train_count = 0
        val_accuracy= 0
        val_count = 0   
        for data,label in train_loader:
            label = label.to(device)-2
            data = data.to(device)
            data = data.unsqueeze(-1).permute(0, 2, 1).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_accuracy += (output.argmax(dim=-1) == label).sum().item()
            train_count += label.size(0)
            train_loss += loss.item()
            outp.append(output)
            inp.append(data)
        #print("Correct: ", train_accuracy, " Total: ", train_count, " Accuracy: ", train_accuracy/train_count, " incorrect: ", train_count-train_accuracy)
        train_accuracies.append(train_accuracy/train_count)
        
        train_loss /= len(train_loader)   
        train_losses.append(train_loss)             
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for dataa, labell in val_loader:
                labell = labell.to(device)
                dataa = dataa.to(device)
                dataa = dataa.unsqueeze(-1).to(device)
                dataa = dataa.permute(0, 2, 1)
                labell = labell -2
                outputt = model(dataa)
                loss = criterion(outputt, labell)
                val_loss += loss.item()
                val_accuracy += (outputt.argmax(dim=-1) == labell).sum().item()
                val_count += labell.size(0)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy/val_count)
        #print("Epoch: " +  str(epoch) + " Training loss: " + str(train_loss) + " Validation loss: " + str(val_loss) + " Validation Accuracy: " + str(val_accuracy/val_count) + " val incorrect: " + str(val_count-val_accuracy))
    torch.save(model.state_dict(), 'modell2.pth')
    torch.save(model, 'model2.pth')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    

    

def autothreshold(losses):
    thresholds = threshold_multiotsu(np.array(losses), classes=2)
    threshol = thresholds[0]
    return threshol

def visualize_histogram_with_threshold(losses, num_bins=256):
    threshold = autothreshold(losses)
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=num_bins, range=(min(losses), max(losses)), density=True, alpha=0.75, color='blue', label='Losses Histogram')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    plt.xlabel('Losses')
    plt.ylabel('Density')
    plt.title('Histogram of Losses with Optimal Threshold')
    plt.legend()
    plt.show()

def test(model, testloader,criterion):
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
        classes = ['Unhealthy','Healthy']
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(ticks=np.arange(2), labels=classes)
        plt.yticks(ticks=np.arange(2), labels=classes)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion[i, j], ha='center', va='center', color='red')
        plt.show()
        print(confusion)
        Precision, Recall, F1sc, sup = precision_recall_fscore_support(labels, predictions, average='binary')
        print("Precision: " + str(Precision), " Recall: " + str(Recall), " F1: " + str(F1sc))
        accuracy = (confusion[0, 0] + confusion[1, 1]) / np.sum(confusion)
        print("Accuracy: " + str(accuracy))
        
def test2(model, testloader):
    model.eval()
    with torch.no_grad():
        labels = []
        predictions = []
        for data, label in testloader:
            data = data.unsqueeze(-1).to(device)
            data = data.permute(0, 2, 1)        
            label = label -2
            label = label.to(device)
            output = model(data) 
            labels.append(label.cpu().numpy())
            predictions.append(torch.argmax(output, dim=1).cpu().numpy())

        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        confusion = confusion_matrix(labels, predictions)
        classes = ['R-T', 'PVC', 'SB/EB', 'FVN']
        plt.matshow(confusion)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(ticks=np.arange(4), labels=classes)
        plt.yticks(ticks=np.arange(4), labels=classes)
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                plt.text(j, i, confusion[i, j], ha='center', va='center', color='red')
        plt.show()
        print(confusion)
        Precision, Recall, F1sc, _ = precision_recall_fscore_support(labels, predictions, average='weighted',zero_division=1)
        print(f"Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1: {F1sc:.4f}")
        accuracy = np.trace(confusion) / np.sum(confusion)
        print(f"Accuracy: {accuracy:.4f}")
        
        
     
def test3(model,model2,test_loader,criterion):
    model.eval()
    model2.eval()
    with torch.no_grad():
        labels = []
        predictions = []
        unhealth_data = []
        unhealth_labels = []
        original_labels = []
        original_data = []
        losses = []
        tmp1 = 0
        tmp2 = 0
        for data, label in test_loader:
            data = data.unsqueeze(-1).to(device)
            output = model(data)
            loss = criterion(output, data)
            losses.append(loss.item())
            if label == 1:#converts labels to 1 being healthy and 0 being unhealthy 
                labels.append(1)#healthy data
                tmp1 += 1#debuging
            else:
                labels.append(0)#unhealthy data
                tmp2 += 1#debuging
            original_labels.append(label.cpu().numpy())
            original_data.append(data.cpu().numpy())
        print("first part: Healthy data: ", tmp1, " Unhealthy data: ", tmp2)
        visualize_histogram_with_threshold(losses)
        optimal_threshold = autothreshold(losses)
        print(f"Optimal Threshold: {optimal_threshold}")
        tmp1 = 0
        tmp2 = 0
        tmp3 = 0
        for i,loss in enumerate(losses):
            if loss < optimal_threshold:
                predictions.append(1)
                tmp1 += 1
            else:
                predictions.append(0)
                tmp2 += 1
                
                if original_labels[i] == 1:
                    tmp3 += 1
                unhealth_data.append(original_data[i])
                unhealth_labels.append(original_labels[i])
                
        print("second part: Healthy data: ", tmp1, " Unhealthy data: ", tmp2, " Unhealthy data that was misclassified: ", tmp3)
        confusion = confusion_matrix(labels, predictions)
        classes = ['Unhealthy','Healthy']
        plt.matshow(confusion)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(ticks=np.arange(2), labels=classes)
        plt.yticks(ticks=np.arange(2), labels=classes)
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                plt.text(j, i, confusion[i, j], ha='center', va='center', color='red')
        plt.show()
        print(confusion)
        Precision, Recall, F1sc, sup = precision_recall_fscore_support(labels, predictions, average='binary')
        print("Precision: " + str(Precision), " Recall: " + str(Recall), " F1: " + str(F1sc))
        accuracy = (confusion[0, 0] + confusion[1, 1]) / np.sum(confusion)
        print("Accuracy: " + str(accuracy))
        labels2 = []
        print("running model2...")
        
        predictions2 = []
        if unhealth_data:
            unhealth_data_tensor = torch.tensor(unhealth_data).to(device)
            unhealth_labels = [label.item() for label in unhealth_labels]
            unhealth_labels_tensor = (torch.tensor(unhealth_labels)-2).to(device)
            unhealth_labels_tensor[unhealth_labels_tensor == -1] = 4
            for i in range(unhealth_data_tensor.size(0)):
                data = unhealth_data_tensor[i].to(device)
                data = data.permute(0, 2, 1)  
                output = model2(data)
                predictions2.append(torch.argmax(output, dim=1).cpu().numpy())
                label = unhealth_labels_tensor[i]
                labels2.append(label.cpu().numpy())
            labels2 = np.array(labels2).flatten()
            predictions2 = np.array(predictions2).flatten()  
            confusion = confusion_matrix(labels2, predictions2)
            plt.matshow(confusion)
            classes = ['R-T', 'PVC', 'SB/EB', 'FVN', 'Healthy']
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.xticks(ticks=np.arange(5), labels=classes)
            plt.yticks(ticks=np.arange(5), labels=classes)
            for i in range(confusion.shape[0]):
                for j in range(confusion.shape[1]):
                    plt.text(j, i, confusion[i, j], ha='center', va='center', color='red')
            plt.show()
            print(confusion)
            Precision, Recall, F1sc, _ = precision_recall_fscore_support(labels2, predictions2, average='weighted',zero_division=1)
            print(f"Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1: {F1sc:.4f}")
            accuracy = np.trace(confusion) / np.sum(confusion)
            print(f"Accuracy: {accuracy:.4f}")

        
        
        
        

usr_inp = input("Choose one of these options: \n1 - Test ECG-NET\n2 - Test ECG-TCN\n3 - Test ECG-NET + ECG-TCN\n4 - Train ECG-Net \n5 - Train ECG-TCN \n")
if usr_inp == '1':
    print("Testing ECG-Net")
    model = torch.load('model.pth')
    criterion = nn.L1Loss(reduction='sum')#sum
    test(model, test_loader,criterion)
elif usr_inp == '2':
    print("Testing ECG-TCN")
    model2 = torch.load('model2.pth')
    test2(model2, test_unh)
elif usr_inp == '3':
    print("Testing ECG-Net + ECG-TCN")
    model = torch.load('model.pth')
    model2 = torch.load('model2.pth')
    criterion = nn.L1Loss(reduction='sum')#sum
    test3(model,model2,test_loader,criterion)
elif usr_inp == '4':
    print("Training ECG-Net")
    epochs = 100
    model = ecgnet()
    model = model.to(device)
    criterion = nn.L1Loss(reduction='sum')#sum
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    train(model, train_loader, val_loader, epochs, criterion, optimizer, device)
elif usr_inp == '5':
    print("Training ECG-TCN")
    model2 = customnet()
    epochs2 = 100
    class_weight = torch.tensor([w0, w1, w2, w3]).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=class_weight)
    model2 = model2.to(device)
    model2.apply(weight_init)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001, weight_decay=0.0001)
    train2(model2, train_unh, val_unh, epochs2, criterion2, optimizer2, device)
else:
    print("Invalid input")




        
