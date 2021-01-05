from Q1_1 import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import pandas as pd

if __name__ == '__main__':
    nn = MyNeuralNetwork(5,(784,256,128,64,10),'linear',0.1,'normal',200,100)
    train_data =pd.read_csv('mnist_train.csv',sep=",",header=None)
    test_data = pd.read_csv('mnist_test.csv',sep=",",header=None)
    # print(train_data)
    # print(test_data)
    from sklearn.model_selection import train_test_split
    # X_train, y_train,a,b = train_test_split(X_train, y_train, test_size=0.80, random_state=42,stratify=y_train)
    # X_test, y_test,a1,b1 = train_test_split(X_test, y_test, test_size=0.80, random_state=42,stratify=y_train)

    X_train = (train_data.iloc[:,1:].values)/255  # to normalise #use stratified split 
    y_train = train_data.iloc[:,0].values
    X_train,ax,y_train,ay=train_test_split(X_train,y_train,test_size=0.8,stratify=y_train)
    y_test = test_data.iloc[:,0].values
    X_test = (test_data.iloc[:,1:].values)
    X_test,ax,y_test,ay=train_test_split(X_test,y_test,test_size=0.8,stratify=y_test)
    y_test = nn.indices_to_one_hot(y_test,10)
    X_train=X_train/X_train.max()
    X_test=X_test/X_test.max()
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    nn.fit(X_train,y_train)
    test_loss=[]
    test_accuracy=[]
    for i in range(100): #100 epochs
      nn.initialise_layers(len(y_test))
      nn.forward(X_test)
      test_loss.append(nn.cross_entropy(nn.output,y_test))
      # test_accuracy.append(self.accuracy(self.output,self.y_test)     
    plt.plot(nn.loss)
    # plt.plot(test_loss)
    print(nn.score(X_test,y_test),"Final score")
    # saving weights using pickle
    import pickle
    pickle.dump(nn,open("tanh.sav",'wb'))
    #plotting curves
    plt.plot(nn.loss[2:],label="train loss")
    plt.plot(test_loss[5:],label="test loss")
    plt.legend()
#     #tsne
# from sklearn.manifold import TSNE
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle
# import pandas as pd

# nn_load = pd.read_pickle(r"/content/gdrive/MyDrive/Dataset/linear1.sav")
# test_data = pd.read_csv('/content/gdrive/MyDrive/Dataset/mnist_test.csv',sep=",",header=None)
# y_test = test_data.iloc[:,0].values
# # print(y_test.shape)
# X_test = (test_data.iloc[:,1:].values)
# X_test,ax,y_test,ay=train_test_split(X_test,y_test,test_size=0.9,stratify=y_test)
# nn_load.forward(X_test)
# tSNE = TSNE(n_components=2)
# print(nn.layer_values[-2].shape)
# X = nn.layer_values[-2]
# X = tSNE.fit_transform(X)
# print(X_test.shape,y_test.shape)
# print(X.shape,"X shape")
# # X['Y'] = y
# plt.figure(figsize=(16,4))
# ax1 = plt.subplot(1, 3, 1)       #plotting the scatterplots to show the segregation of the datapoints
# sns.scatterplot(
#     x=X.T[0], y=X.T[1],
#     hue=y_test,
#     # palette=sns.color_palette("hls", 10),
#     # data=X_test,
#     # legend="full",
#     # alpha=0.3,
#     # ax=ax1
# )
# plt.show()