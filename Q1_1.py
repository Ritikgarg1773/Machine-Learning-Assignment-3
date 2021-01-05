import matplotlib.pyplot as plt
import numpy as np
import random
#reference: https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9

class MyNeuralNetwork():# class for neural network
	acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
	weight_inits = ['zero', 'random', 'normal']

	def __init__(self, n_layers, layer_sizes, activation='relu', learning_rate=0.1, weight_init='normal', batch_size=50, num_epochs=100):
		#initialise the global variables and values are by default set if not given in the constructor
		if activation not in self.acti_fns:
			raise Exception('Incorrect Activation Function')

		if weight_init not in self.weight_inits:
			raise Exception('Incorrect Weight Initialization Function')

		self.layers=n_layers
		self.layer_sizes=layer_sizes
		self.act_func=activation
		self.learning_rate=learning_rate
		self.weight_init=weight_init
		self.batch_size=batch_size
		self.epochs=num_epochs
		self.weights=[[]] #dummy layer added because the input layer acts as just a dummy layer
		self.biases=[[]]  #dummy layer added
		for  i in range(self.layers-1):
			bias= self.zero_init((self.layer_sizes[i+1]))
			weight=self.normal_init((self.layer_sizes[i],self.layer_sizes[i+1]))
			self.weights.append(weight)
			self.biases.append(bias)
		# arrays to store the values of bias and weight after each epoch
		self.itr_weights=[]
		self.itr_biases=[]


	def relu(self, X):
		ans=np.copy(X)
		return np.maximum(ans,0)

	def relu_grad(self, X):
		ans=np.copy(X)
		ans[ans>=0]=1
		ans[ans<0]=0
		return ans

	def sigmoid(self, X):
		ans=np.copy(X)
		return 1/(1+np.exp(-ans))

	def sigmoid_grad(self, X):
		ans=np.copy(X)
		sig=self.sigmoid(X)
		return sig*(1-sig)

	def linear(self, X):
		ans=np.copy(X)
		return ans

	def linear_grad(self, X):
		return np.ones(X.shape)

	def tanh(self, X):
		ans=np.copy(X)
		return 2*self.sigmoid(2*ans)-1

	def tanh_grad(self, X):
		ans=np.copy(X)
		return 1- np.power(self.tanh(ans),2)

	def softmax(self, X):
		ans=np.copy(X)
		a=np.exp(ans)
		b=np.sum(a,axis=-1).reshape(a.shape[0],1)
		return a/b

	def softmax_grad(self, X):
		ans=np.copy(X)
		b=np.sum(np.exp(ans),axis=-1)
		b=b.reshape(b.shape[0],1)
		a=np.exp(ans)*(b-np.exp(ans))
		return a/np.power(b,2)

	def zero_init(self, shape):
		return np.zeros(shape)

	def random_init(self, shape):
		np.random.seed(69)
		return 0.01*np.random.rand(shape)

	def normal_init(self, shape):
		np.random.seed(69)
		return 0.01*np.random.normal(0,1,shape)

	def apply_act_func(self,X):
		#for the chosen activation function, it applies the corresponding activaation function and returns it
		X=np.copy(X)
		if(self.act_func=='relu'):
			return self.relu(X)
		elif(self.act_func=='sigmoid'):
			return self.sigmoid(X)
		elif(self.act_func=='linear'):
			return self.linear(X)
		elif(self.act_func=='tanh'):
			return self.tanh(X)
		else:
			return self.softmax(X)

	def apply_act_func_grad(self,X):
		#for the chosen activation function, it applies the corresponding activaation function' gradient and returns it
		X=np.copy(X)
		if(self.act_func=='relu'):
			return self.relu_grad(X)
		elif(self.act_func=='sigmoid'):
			return self.sigmoid_grad(X)
		elif(self.act_func=='linear'):
			return self.linear_grad(X)
		elif(self.act_func=='tanh'):
			return self.tanh_grad(X)
		else:
			return self.softmax_grad(X)

	def fit(self, X, y):
		#model training
		self.X_train=np.copy(X)
		self.Y_train=np.copy(y)
		num_batches=len(X)//self.batch_size #number of batches are computed for the given batch sizze
		for i in range(self.epochs):
			for j in range(num_batches):
				#creating batch from whole data
				start=j*self.batch_size
				temp_X=np.copy(self.X_train[start:start+self.batch_size,:])
				temp_Y=np.copy(self.Y_train[start:start+self.batch_size])
				#forward phase
				output=self.forward(temp_X)
				#Backward phase
				self.backward(temp_X,temp_Y,output)
			#storing wieghts and biases for each epoch
			self.itr_weights.append(np.copy(self.weights))
			self.itr_biases.append(np.copy(self.biases))

		return self

	def forward(self,X):
		#this function controls the forward propagation flow 
		input_vector=np.copy(X)
		self.y_before=[[]]#array to store wi*xi_bi for each layer i.e. output before applying activation function 
		self.y_after=[[]]##array to store output after applying activation function on wi*xi+bi
		for i in range(1,self.layers):
			try:
				output1=np.matmul(input_vector,self.weights[i])+self.biases[i]
				self.y_before.append(output1)
				if(i==self.layers-1):
					output2=self.softmax(output1)
				else:
					output2=self.apply_act_func(output1)
				self.y_after.append(output2)
				input_vector=output2
			except:
				print("layer=",i,"  ",input_vector.shape,self.weights[i].shape,self.biases[i].shape)

		return(input_vector)

	def update_weight(self,layer_no,weight_gradient):
		#updates the value of weights of a given layer
		self.weights[layer_no] = self.weights[layer_no] - self.learning_rate * weight_gradient
	def update_bias(self,layer_no,bias_gradient):
		#updates the value of biases of a given layer
		self.biases[layer_no] = self.biases[layer_no] - self.learning_rate * bias_gradient

	def backward(self, X, y, output):
		# this controls the back propagation flow of the nn
		#making array of ones corresponding to each class value in place of the probability to calculate loss and the gradient of cross entropy
		array_of_ones = np.zeros_like(output)
		array_of_ones[np.arange(len(output)),y] = 1
		#calculating (1-p) for nn's last layer's output
		middle = (- array_of_ones + output) / output.shape[0]
		input_gradient = np.dot(middle, self.weights[self.layers-1].T)
		weight_gradient = np.dot(self.y_after[self.layers-2].T, middle)
		bias_gradient = middle.mean(axis=0)*self.y_after[self.layers-2].shape[0]
		##updating weights and bias of last layer
		self.update_weight(self.layers-1,weight_gradient)
		self.update_bias(self.layers-1,bias_gradient)
		
		for layer in range(self.layers-2,0,-1):
			out_grad= self.apply_act_func_grad(self.y_before[layer])
			loss_by_sum = input_gradient * out_grad
			loss_by_out = np.dot(loss_by_sum,self.weights[layer].T)

			if(layer==1):#for the first hidden layer we use features give in X
				loss_by_weight = np.dot(X.T,loss_by_sum)
				loss_by_bias = loss_by_sum.mean(axis=0)*X.shape[0]
			else:
				loss_by_weight = np.dot(self.y_after[layer-1].T,loss_by_sum) 
				loss_by_bias = loss_by_sum.mean(axis=0)*self.y_after[layer-1].shape[0]
			input_gradient = loss_by_out
			##updating weights and bias of layer iterating on
			self.update_weight(layer,loss_by_weight)
			self.update_bias(layer,loss_by_bias)
			
	
	def predict_proba(self, X):
		#returns the probabilities of each class for given features x
		out=self.forward(X)
		return out

	def predict(self, X):
		#returns the predicted class based on the highest values of probabilities calculated for each class
		out=self.predict_proba(X)
		return np.argmax(out,axis=1)

	def score(self, X, y):
		#calculates the score cosseponding to the prediction done on given X
		y_pred=self.predict(X)
		return np.sum(y_pred==y)/y.shape[0]

	def cross_entropy_loss(self, out, y):
		#returns the cross entropy loss
		output=[]
		for i in range(len(y)):
			output.append(out[i,y[i]])
		output=np.array(output)
		error = -(np.log(output+(10**(-12) )))
		return np.mean(error)

	def plot(self,X_train,X_test,Y_train,Y_test):
		#plots the training loss and testing loss corresponding to each epochs
		#computing training loss for eac epoch one by one after doing forward prop with the weights of that epoch 
		training_loss = []
		for i in range(self.epochs):
			input_vector=X_train
			weight=self.itr_weights[i]
			biases=self.itr_biases[i]
			for i in range(1,self.layers):
				output1=np.matmul(input_vector,weight[i])+biases[i]
				if(i==self.layers-1):
					output2=self.softmax(output1)
				else:
					output2=self.apply_act_func(output1)
				input_vector=output2

			loss_val=self.cross_entropy_loss(input_vector,Y_train)
			training_loss.append(loss_val)

		#computing testing loss for eac epoch one by one after doing forward prop with the weights of that epoch 
		testing_loss = []
		for i in range(self.epochs):
			input_vector=X_test
			weight=self.itr_weights[i]
			biases=self.itr_biases[i]
			for i in range(1,self.layers):
				output1=np.matmul(input_vector,weight[i])+biases[i]
				if(i==self.layers-1):
					output2=self.softmax(output1)
				else:
					output2=self.apply_act_func(output1)
				input_vector=output2

			loss_val=self.cross_entropy_loss(input_vector,Y_test)
			testing_loss.append(loss_val)

		#plotting 
		plt.plot(training_loss,label='Train error')
		plt.plot(testing_loss,label='Test error')
		plt.ylabel('Cross Entropy Error')
		plt.xlabel('Epochs')
		plt.legend()
		plt.show()
