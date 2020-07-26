import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import Model, ConvNet
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import sys
writer = SummaryWriter("runs/mnist")
# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
input_size = 784 # 28 * 28
hidden_size1 = 256
hidden_size2 = 128
num_classes = 10 # 0-9
learning_rate = 0.0001
batch_size = 100
num_epochs = 20

# step 0: prepare data
class TrainDataset(Dataset):
    def __init__(self, file_name):
        xy = np.loadtxt(file_name, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.x = self.x.view(-1, 1, 28, 28)
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
        self.x = self.x.view(-1, 1, 28, 28)
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
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     print(features.shape)
#     plt.imshow(features[i][0], cmap = 'gray')
# plt.show()
img_grid = torchvision.utils.make_grid(features)
writer.add_image('mnist images', img_grid)


# datiter = iter(test_loader)
# features = datiter.next()
# print(features.shape)
print('Dataset is ready')

# Step 1 : model
# model = Model(input_size=input_size, hidden_size1=hidden_size1, hidden_size2 = hidden_size2, num_classes=num_classes).to(device)
model = ConvNet().to(device)
print('Model is declared')

# Step 2 : loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, features)
# writer.close()
# sys.exit(0)
# step 3: training
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device) # cpu or gpu
        labels = labels.to(device) # cpu or gpu

        #forward pass
        y_predict = model(features)
        loss = criterion(y_predict, labels)

        # backward pass

        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(y_predict, 1)
        running_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step: [{i}/{n_total_steps}], Loss: [{loss.item():0.4f}]')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0
file_name = 'model.pth'
torch.save(model, file_name)
print(f'Training is complete and model is saved in {file_name}')
# step 4: evaluate
class_labels = []
class_preds = []
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

        class_prob_batch = [F.softmax(output, dim=0) for output in outputs]

        class_preds.append(class_prob_batch) # 10 different class probability
        class_labels.append(predictions) # single class prediction

class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
class_labels = torch.cat(class_labels)
acc = 100.0 * n_correct / n_total
print(f'Accuracy for {n_total} images is {acc:0.4f}')
#tensorboard
for i in range(10):
    label_i = class_labels == i
    preds_i = class_preds[:, i]
    writer.add_pr_curve(str(i), label_i, preds_i, global_step=0)
    writer.close()
#evaluation in test set:

image_id = []
label = []

y_pred = model(test_dataset.x).detach()
_, predictions = torch.max(y_pred, 1)
for i, pred in enumerate(predictions):
    image_id.append(i + 1)
    pred = pred.item()
    label.append(pred)
submission_dict = {
    'ImageId': image_id,
    'Label': label
}
df = pd.DataFrame(submission_dict)
df.to_csv('submission_v1.csv', index=False)