import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import PimaClassifier


# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


# Initialize the model
model = PimaClassifier()

# Train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 1000
batch_size = 10
 
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Compute the accuracy
with torch.no_grad():
    y_pred = model(X)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y).float().mean()
    print("Accuracy: %.2f" % (accuracy.item() * 100))


# Save the model
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model.pt') # Save
