import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from process_data import unpickle, dataloader
import torch.optim as optim
from CNN import ConvNet

CIFAR_DIR = 'dataset/cifar-10-batches-py/'

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0, 1, 2, 3, 4, 5, 6]

for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)


batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

batch_1 = data_batch1
batch_2 = data_batch2
batch_3 = data_batch3
batch_4 = data_batch4
batch_5 = data_batch5
test_batch = test_batch

data_1, label_1 = data_batch1[b'data'].reshape((10000, 3, 32, 32)), batch_1[b'labels']
data_2, label_2 = data_batch2[b'data'].reshape((10000, 3, 32, 32)), batch_2[b'labels']
data_3, label_3 = data_batch3[b'data'].reshape((10000, 3, 32, 32)), batch_3[b'labels']
data_4, label_4 = data_batch4[b'data'].reshape((10000, 3, 32, 32)), batch_4[b'labels']
data_5, label_5 = data_batch5[b'data'].reshape((10000, 3, 32, 32)), batch_5[b'labels']
test_data, test_labels = test_batch[b'data'].reshape((10000, 3, 32, 32)), test_batch[b'labels']

#Training the model(Hyperparameters)
num_epochs = 20 
model = ConvNet().to(device)
batch_size = 100
learning_rate = 0.001

#define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#dataset has PILImage images of range [0,1]
#We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



train_arr = np.concatenate((data_1, data_2, data_3, data_4, data_5), axis=0)
train_arr = train_arr.transpose(0, 2, 3, 1).astype(np.float32)
train_labels = label_1 + label_2 + label_3 + label_4 + label_5
train_labels = np.array(train_labels)
test_data = test_data.transpose(0, 2, 3, 1).astype(np.float32)
test_labels = np.array(test_labels)


train_labels = torch.LongTensor(train_labels)
train_data = dataloader(train_arr, train_labels)
train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

test_data = dataloader(test_data, test_labels)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


n_total_steps = len(train_loader)
# n_total_steps = number of batches i.e 50,000/4 where 4 is the batch size

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #original shape: [4, 3, 32, 32] = 4, 3, 1024
        images = images.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 ==0:
            print (f'Epoch [{epoch+1}/{num_epochs}], step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
print('Finished Training')

#Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range (10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        #max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
