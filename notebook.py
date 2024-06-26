#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Road Map

# X Step 1  - EDA
# ✓ Step 2  - Create our Custom Dataset with Labels
# ✓ Step 3  - Split DataFrame into Test and Train
# ✓ Step 4  - Create Project 2 CNN Model
# ✓ Step 5  - Running a Test CNN to ensure model functions over 10 epochs
# ✓ Step 6  - Use Optuna Optimizer on our CNN model and apply parameters over 100 epochs
# ✓ Step 7  - Validate the Final model 10 times using 150 epochs
# ✓ Step 8  - Run the final model over 500 epochs
# X Step 9  - Evaluate with "suitable metrics" (Confusion Matrix, AUCROC)
# X Step 10 - Save the Model
# X Step 11 - Write the technical report
# X Step 12 - Compile all documents


# In[2]:


# Put all packages here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import statistics
import scipy.stats as st
import datetime
from IPython.display import clear_output


# In[3]:


#Declaring all Project 2 Variables

# Dataset Variables
sizeClasses = 2
sizeBatch = 32
sizeImage = 256

# Image Size
imageHeight = 7
imageWidth = 7
imageTransHeight = 9
imageTransWidth = 9

learningRate = 0.001
testSize = 0.2
randomState = 42

# CNN Variables
sizeKernel = 3
sizeStride = 1
sizePadding = 1
sizePoolKernel = 2
sizePoolStride = 2
testEpochs = 10
tuneEpochs = 100
validateEpochs = 150
finalEpochs = 500

# Import Data Sets
arrayLions = []
arrayCheetahs = []
folderLions = ".\\images\\Lions"
folderCheetahs = ".\\images\\Cheetahs"
folderRoot = ".\\images"

# K-Fold Validation
nSplits = 10


# In[4]:


# Creating the DataFrame

# Importing Lions
for r,d,f in os.walk(folderLions):
    for file in f:
        arrayLions.append((os.path.join(folderLions, file), "0"))
    
# Importing Cheetahs
for r,d,f in os.walk(folderCheetahs):
    for file in f:
        arrayCheetahs.append((os.path.join(folderCheetahs, file), "1"))
        
# # Combining Numpy Arrays
arrayCombined = np.concatenate((arrayLions, arrayCheetahs), axis = 0)

# # Creating the DataFrame
df = pd.DataFrame(arrayCombined, columns = ['name', 'label'])


# In[5]:


# Apply Labels to the DataFrame

# Setting the Transforms
transform = transforms.Compose([
                                transforms.Resize([imageTransHeight, imageTransWidth]),
                                transforms.ToTensor(),
                                ])

# Label Mapping
mappingLabel = {'0': 0, '1': 1}

# Creating proj2CNN Class
class LCDataset(Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.fileName = df['name'].values
        self.labels = df['label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        pathImage = self.fileName[index]
        image = Image.open(pathImage).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(mappingLabel[self.labels[index]],
                            dtype = torch.long)
        return image, label
    
# Loading the New Dataset
newDataset = LCDataset(df, transform = transform)


# In[6]:


# PERFORM EDA IN HERE
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


# In[7]:


# Creating the Test and Train Sets and DataLoaders

# Creating Train and Test Sets
setTrain, setTest = random_split(newDataset, [0.8, 0.2])

# Loading Datasets
trainLoader = torch.utils.data.DataLoader(setTrain, batch_size = sizeBatch, shuffle = True)
testLoader = torch.utils.data.DataLoader(setTest, batch_size = sizeBatch, shuffle = True)


# In[8]:


# Creating the CNN for Project 2

class proj2CNN(nn.Module):
    def __init__(self, sizeKernel, sizeStride, sizePadding, sizePoolKernel, sizePoolStride, imageHeight, imageWidth):
        super(proj2CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 1 input channel, 32 output channels, 3x3 kernel, padding=1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32 input channels, 64 output channels, 3x3 kernel, padding=1
        self.pool = nn.MaxPool2d(2, 2)  # Max-pooling layer with 2x2 kernel
        self.fc1 = nn.Linear(64 * 2 * 2, 128)  # Fully connected layer with 128 units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # Output layer with 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
#         This is used for finding the values for te view
#         print(x.size())
        x = x.view(-1, 64 * 2 * 2)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[9]:


# Setting initial parameters for the test Model

# Creating an instance of proj2CNN
proj2CNNetwork = proj2CNN(sizeKernel, sizeStride, sizePadding, sizePoolKernel, sizePoolStride, imageHeight, imageWidth)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(proj2CNNetwork.parameters(), lr = learningRate)


# In[10]:


# Running the initial test of the CNN Model and initial validation

# Setting the traingLosses array for data storage
trainingLosses = []

#-----

# # Runs the models for the number of desired Epochs
# This validates once after the training is compelte
for epoch in range(testEpochs):
    trainingLoss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = proj2CNNetwork(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        trainingLoss += loss.item()
    trainingLosses.append(trainingLoss / len(trainLoader))
    print(f"Epoch {epoch+1}, Loss: {trainingLoss / len(trainLoader)}")
print("Testing Model Completed")

correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        outputs = proj2CNNetwork(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy on the test dataset: {100 * correct / total}%")


# In[11]:


# Plotting First Training
plt.plot(trainingLosses)


# In[12]:


# Preparing the Optuna for HyperTuning Parameters

def tuning(trial):
    # Setting up the parameters
    learning_rate = trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001, 0.0001])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64] )
    num_channels = trial.suggest_categorical('num_channels', [16, 32, 64])
    
    # Setting up the new Proj2CNNetwork for each parameter trial
    optunaModel = proj2CNN(sizeKernel, sizeStride, sizePadding, sizePoolKernel, sizePoolStride, imageHeight, imageWidth)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(proj2CNNetwork.parameters(), lr = learningRate)

    # Training the model with the testing parameters
    for epoch in range(tuneEpochs):
        trainingLoss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = optunaModel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            trainingLoss += loss.item()
            
        # Caculating the training loss from each epoch
        trainingLoss /= len(trainLoader)
        
    # Calculate accuracy on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data
            outputs = optunaModel(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total

    # Return the negative accuracy as Optuna tries to minimize the objective
    return -accuracy  


# In[13]:


# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(tuning, n_trials=10)


# In[14]:


# Applying Best Parameters

# Get the best hyperparameters
best_params = study.best_params

# Setting the new optimized parameters
learningRate = best_params['learning_rate']
sizeBatch = best_params['batch_size']

# Refreshing the Model
proj2CNNetwork = proj2CNN(sizeKernel, sizeStride, sizePadding, sizePoolKernel, sizePoolStride, imageHeight, imageWidth)


# In[ ]:





# In[15]:


# Validating the Hypertuned Model

# Resetting Validation Loop
splitCount = 0
i = 0

# Setting up k-fold validation
cv = StratifiedKFold(n_splits = nSplits, shuffle = True, random_state = randomState)

# Defining array for recoding the scores
cnn_array = []

while i < 1:
    # Count for splits for progress tracking
    splitCount = 0
    # Looping through the 10 Folds
    for fold, (train_index, val_index) in enumerate(cv.split(X=np.zeros(len(newDataset)), y=newDataset.labels)):
        # Reloading Datasets with tuned batch sizes
        trainLoader = torch.utils.data.DataLoader(setTrain, batch_size = sizeBatch, shuffle = True)
        testLoader = torch.utils.data.DataLoader(setTest, batch_size = sizeBatch, shuffle = True)

        # Reloading optimizer with tuned learning rate
        optimizer = optim.Adam(proj2CNNetwork.parameters(), lr = learningRate)
        
        # Reloading the model for each new fold
        valProj2CNNetwork = proj2CNN(sizeKernel, sizeStride, sizePadding, sizePoolKernel, sizePoolStride, imageHeight, imageWidth)

        for epoch in range(validateEpochs):
            trainingLoss = 0.0
            for i, data in enumerate(trainLoader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = valProj2CNNetwork(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                trainingLoss += loss.item()
            trainingLosses.append(trainingLoss / len(trainLoader))

            # Calculating the training loss from each epoch
            trainingLoss /= len(trainLoader)
            
            # Begin evaluating the model
            valProj2CNNetwork.eval()
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in testLoader:
                    outputs = valProj2CNNetwork(inputs)
                    _, predictions = torch.max(outputs, 1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            clear_output()
            print(f"Fold {fold + 1}, Epoch {epoch + 1} / {validateEpochs}")    
                
            # Calculate accuracy for this fold
            accuracy = accuracy_score(all_labels, all_predictions)
            cnn_array.append(accuracy)
            
        # Visual Confirmation of Fold Completed
        splitCount += 1
        e = datetime.datetime.now()
        print(f"Fold {fold + 1}, Accuracy: {accuracy} at {e.hour}:{e.minute}:{e.second}")

    # Iterating the next epoch
    i += 1   
    
# Final Visual Confirmation for Validation    
print("Complete")
print()

# Finding metrics
print(f"The mean of 10 proj2CNNetwork models is:   {sum(cnn_array)/len(cnn_array)}")
print(f"The stdDev of 10 proj2CNNetwork models is: {statistics.stdev(cnn_array)}")
print(f"The 95% Confidence interval of proj2CNNetwork models is: {st.t.interval(0.95, df=len(cnn_array)-1, loc=np.mean(cnn_array), scale=st.sem(cnn_array))}")


# In[16]:


# Reloading Datasets with tuned batch sizes
trainLoader = torch.utils.data.DataLoader(setTrain, batch_size = sizeBatch, shuffle = True)
testLoader = torch.utils.data.DataLoader(setTest, batch_size = sizeBatch, shuffle = True)

#Reloading optimizer with tuned learning rate
optimizer = optim.Adam(proj2CNNetwork.parameters(), lr = learningRate)

for epoch in range(finalEpochs):
    trainingLoss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = proj2CNNetwork(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        trainingLoss += loss.item()
    trainingLosses.append(trainingLoss / len(trainLoader))
    print(f"Epoch {epoch+1}, Loss: {trainingLoss / len(trainLoader)}")
print("Final Model Completed")

# # Validating the tested model
correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        outputs = proj2CNNetwork(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy on the test dataset: {100 * correct / total}%")


# In[17]:


# Suitable Metrics on final model here?
#
# AUCROC Curve?
#
# Confusion Matrix?
#
#
#
#
#
#
#
#
#
#
#
#
#
#


# In[18]:


# Saving the model for the Webpage
torch.save(proj2CNNetwork.state_dict(),".\\Website\\Proj2.py")

print("Model Saved Succesfully") 


# In[33]:


import pickle

# Save the entire trained model as a pickled file
with open('model.pkl', 'wb') as f:
    pickle.dump(proj2CNNetwork, f)

print("Trained Model Saved Successfully as model.pkl")


# In[34]:


import torch
import pickle

# Save the entire trained model as a pickled file
torch.save(proj2CNNetwork, 'trained_model.pth')

print("Trained Model Saved Successfully as trained_model.pth")


# In[32]:





# In[ ]:




