import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)



# The model expects rows of data with 8 variables (the first argument at the first layer set to 8)
# The first hidden layer has 12 neurons, followed by a ReLU activation function
# The second hidden layer has 8 neurons, followed by another ReLU activation function
# The output layer has one neuron, followed by a sigmoid activation function

model = nn.Sequential(
    nn.Linear(8, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)



# OR:

# class PimaClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden1 = nn.Linear(8, 12)
#         self.act1 = nn.ReLU()
#         self.hidden2 = nn.Linear(12, 8)
#         self.act2 = nn.ReLU()
#         self.output = nn.Linear(8, 1)
#         self.act_output = nn.Sigmoid()
 
#     def forward(self, x):
#         x = self.act1(self.hidden1(x))
#         x = self.act2(self.hidden2(x))
#         x = self.act_output(self.output(x))
#         return x
 
# model = PimaClassifier()
# print(model)



loss_fn = nn.BCELoss()  # binary cross entropy
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)


accuracy_target = 0.99
max_epochs = 1000
batch_size = 10
current_epoch = 0
accuracy = 0.0

while accuracy < accuracy_target and current_epoch < max_epochs:
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    current_epoch += 1
    with torch.no_grad():
        y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Epoch {current_epoch} accuracy {accuracy}")

# compute accuracy (no_grad is optional)

print(f"Accuracy {accuracy}")



 
# make probability predictions with the model
predictions = model(X)
# round predictions
rounded = predictions.round()


# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

