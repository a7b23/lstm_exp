import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import re
import numpy as np

def normalizeString(s):
	s = s.lower().strip()
	s = re.sub(r"<br />",r" ",s)
	s = re.sub(r'(\W)(?=\1)', '', s)
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	
	return s

class Model(torch.nn.Module) :
	def __init__(self,embedding_dim,hidden_dim) :
		super(Model,self).__init__()
		self.hidden_dim = hidden_dim
		self.embeddings = nn.Embedding(vocabLimit+1, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim,hidden_dim)
		self.linearOut = nn.Linear(hidden_dim,2)
	def forward(self,inputs,hidden) :
		x = self.embeddings(inputs).view(len(inputs),1,-1)
		lstm_out,lstm_h = self.lstm(x,hidden)
		x = lstm_out[-1]
		x = self.linearOut(x)
		x = F.log_softmax(x)
		return x,lstm_h
	def init_hidden(self) :
		return (Variable(torch.zeros(1, 1, self.hidden_dim)),Variable(torch.zeros(1, 1, self.hidden_dim)))	




vocabLimit = 50000
max_sequence_len = 500
model = Model(50,100)



with open('dict.pkl','rb') as f :
	word_dict = pickle.load(f)

f = open('labeledTrainData.tsv').readlines()

for i,lines in enumerate(f) :
	if i > 0 and len(lines.split(' ')) > 200 and len(lines.split(' ')) < 250 :
		idx = i
		break

print idx
for i in range(4) :
	f1 = open('data_c'+str(i)+'.txt','w')
	model.load_state_dict(torch.load('model'+str(i)+'.pth'))
	data = normalizeString(f[idx].split('\t')[2]).strip()
	input_data = [word_dict[word] for word in data.split(' ')]
	# input_data = Variable(torch.LongTensor(input_data))
	hidden = model.init_hidden()
	for j in range(len(input_data)) :
		temp = hidden[1].data.numpy()
		y_pred,hidden = model(Variable(torch.LongTensor([input_data[j]])),hidden)
		# if j== 0  :
			# print hidden[0].data.numpy()

		f1.write(str(np.mean(abs(hidden[1].data.numpy()-temp)))+'\n')

	# y_pred,_ = model(input_data,hidden)
	print y_pred
	f1.close()

# print y_pred		
# data = f[2].split('\t')[2]
# data = normalizeString(data).strip()
# input_data = [word_dict[word] for word in data.split(' ')]
# if len(input_data) > max_sequence_len :
# 	input_data = input_data[0:max_sequence_len]

# input_data = Variable(torch.LongTensor(input_data))

# hidden = model.init_hidden()
# y_pred,_ = model(input_data,hidden)

# print y_pred	