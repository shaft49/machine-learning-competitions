import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from model import Model
# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
input_size = 784 # 28 * 28
hidden_size = 100
num_classes = 10 # 0-9
learning_rate = 0.0001
batch_size = 100
num_epochs = 45

# step 0: prepare data
class TrainDataset(Dataset):
    def __init__(self, file_name):
        xy = np.loadtxt(file_name, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0].astype(np.long))
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

class TestDataset(Dataset):
    def __init__(self, file_name):
        xy = np.loadtxt(file_name, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return self.n_samples

print('Preparing Datasets')
train_dataset = TrainDataset('train.csv')
test_dataset = TestDataset('test.csv')
first_data = train_dataset[0]
features, labels = first_data
print(features.shape, labels)

#Data loader
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

datiter = iter(train_loader)
features, labels = datiter.next()
print(features.shape, labels.shape)

datiter = iter(test_loader)
features = datiter.next()
print(features.shape)
print('Dataset is ready')

# Step 1 : model
model = Model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
print('Model is declared')

# Step 2 : loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# step 3: training
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device) # cpu or gpu
        labels = labels.to(device) # cpu or gpu

        #forward pass
        y_predict = model(features)
        loss = criterion(y_predict, labels)

        # backward pass
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step: [{i}/{n_total_steps}], Loss: [{loss.item():0.4f}]')
print('Training is complete')
# step 4: evaluate
with torch.no_grad():
    n_correct = 0
    n_total = 0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features)

        _, predictions = torch.max(outputs, 1)
        n_total += labels.size(0)
        n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct / n_total
print(f'Accuracy for {n_total} images is {acc:0.4f}')