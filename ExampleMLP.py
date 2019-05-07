import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gzip
import pickle
import librosa

pickle_off=gzip.open("drive/My Drive/COMP562/patches.data","rb")
emp=pickle.load(pickle_off)

sampling_rate=44100
nsamples=1000;
nmfccs=3;
ntrain=600;
nfeatures=432*(nmfccs+1)

X=torch.empty(nsamples,nfeatures)
y=torch.empty(nsamples,255)

for i in range(nsamples):
    #toneTemp=librosa.feature.tonnetz(y=np.array(emp[1][0]), sr=sampling_rate).flatten()
    rollTemp=librosa.feature.spectral_rolloff(y=np.array(emp[i][0]), sr=sampling_rate).flatten()
    mfccTemp=librosa.feature.mfcc(y=np.array(emp[i][0]), sr=sampling_rate,n_mfcc=nmfccs).flatten()
    merged=np.concatenate((mfccTemp,rollTemp))
    X[i,:]=torch.tensor(merged)
    y[i,:]=torch.tensor([emp[i][1][j][1] for j in range(0,255)])

testIndex=set(np.random.choice(1000,size=ntrain,replace=False))
trainIndex=set(np.linspace(0,999,1000))-testIndex
X_train=X[list(trainIndex),:]
y_train=y[list(trainIndex),:]
X_test=X[list(testIndex),:]
y_test=y[list(testIndex),:]

class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
		self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden5 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden6 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden7 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden8 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden9 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden10 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden11 = torch.nn.Linear(n_hidden, n_hidden)
		self.hidden12 = torch.nn.Linear(n_hidden, n_hidden)
		self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
	def forward(self, x):
		x = F.relu(self.hidden1(x))      # activation function for hidden layer
		x = F.relu(self.hidden2(x))
		x = torch.tanh(self.hidden3(x))
		x = F.relu(self.hidden4(x))
		x = F.relu(self.hidden5(x))
		x = F.relu(self.hidden6(x))
		x = F.relu(self.hidden7(x))
		x = F.relu(self.hidden8(x))
		x = F.relu(self.hidden9(x))
		x = torch.tanh(self.hidden10(x))
		x = F.relu(self.hidden11(x))
		x = F.relu(self.hidden12(x))
		x = self.predict(x)             # linear output
		return x
		
net = Net(n_feature=nfeatures, n_hidden=150, n_output=255)     # define the network
net.cuda()
print(net)  # net architecture

optimizer = torch.optim.RMSprop(net.parameters(), lr=0.00001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

for t in range(200000):
	prediction = net(X_train.cuda())     # input x and predict based on x
	loss = torch.sqrt(loss_func(prediction.cuda(), y_train.cuda()))     # must be (1. nn output, 2. target)
	optimizer.zero_grad()   # clear gradients for next train
	loss.backward()         # backpropagation, compute gradients
	optimizer.step()        # apply gradients
	if t % 5000 == 0:
		# plot and show learning process
		print('Loss=%.4f' % loss.cpu().data.numpy())
    
torch.save(net.state_dict(),"drive/My Drive/COMP562/DeepMLPMFCC_Roll150")

diff=net(X_test.cuda()).cpu()-y_test
diff=torch.pow(diff,2)
dim=list(diff.shape)
torch.sqrt(diff.sum(1).sum(0)/(dim[0]*dim[1]))