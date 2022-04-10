"""训练"""
import time
import torch
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import imageio
from sklearn.decomposition import PCA 


class twoNN:
	"""一个2层神经网络,即输入-隐藏层-输出"""

	def __init__(self, hidden_size, lam, input_size, output_size, weight_init_std = 0.01):
		# 初始化网络
		self.params = {}
		# weight_init_std:权重初始化标准差
		self.params['W1'] = weight_init_std * \
							np.random.randn(input_size, hidden_size) 
							# 用高斯分布随机初始化一个权重参数矩阵
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * \
							np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)
		self.lam = lam
		self.train_loss_list = []
		self.test_loss_list = []
		self.train_acc_list = []
		self.test_acc_list = []


	def predict(self, x):
		# 前向传播
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']

		a1 = np.dot(x, W1) + b1
		z1 = 1 / (1 + np.exp(-a1))
		y = np.dot(z1, W2) + b2

		return y


	def loss(self, x, t):
		# total loss = data loss + regularization
		y = self.predict(x)
		data_loss = np.square(y-t).sum() / (2 * y.shape[0])
		l2_reg = (self.lam * np.square(self.params['W1']).sum() + \
				 self.lam * np.square(self.params['W2']).sum()) / 2

		return data_loss + l2_reg


	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)

		accuracy = np.sum( y==t ) / float(x.shape[0])
		return accuracy


	def gradient(self, x, t):
		# 利用反向传播计算梯度
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']
		grads = {}

		# 前向
		a1 = np.dot(x, W1) + b1
		z1 = 1 / (1 + np.exp(-a1))
		y = np.dot(z1, W2) + b2

		# 反向
		dy = (y - t) / y.shape[0]
		grads['W2'] = np.dot(z1.T, dy) + self.lam * W2
		grads['b2'] = np.sum(dy, axis=0)
		dz1 = np.dot(dy, W2.T)
		da1 = (1.0 - z1) * z1 * dz1
		grads['W1'] = np.dot(x.T, da1) + self.lam * W1
		grads['b1'] = np.sum(da1, axis=0)

		return grads



def training(learning_rate, hidden_size, lam, x_train, \
				t_train, x_test, t_test, iters_num, batch_size):


		train_size = x_train.shape[0]
		iter_per_epoch = max(train_size / batch_size, 1)

		network = twoNN(hidden_size, lam, input_size=784, output_size=10)

		for i in range(iters_num):
			# 获取mini-batch
			batch_mask = np.random.choice(train_size, batch_size)
			x_batch = x_train[batch_mask]
			t_batch = t_train[batch_mask]

			# 学习率下降策略(等步长调整: 每隔1个epoch，lr = lr * 0.95)
			if i % iter_per_epoch == 0:
				learning_rate = learning_rate * 0.95

			# SGD计算梯度
			grad = network.gradient(x_batch, t_batch)

			# SGD更新参数
			for key in ('W1', 'b1', 'W2', 'b2'):
				network.params[key] -= learning_rate * grad[key]

			# 记录学习过程的损失变化
			if i % iter_per_epoch == 0:
				train_loss = network.loss(x_batch, t_batch)
				network.train_loss_list.append(train_loss)
				test_loss = network.loss(x_test, t_test)
				network.test_loss_list.append(test_loss)


			if i % iter_per_epoch == 0:
				train_acc = network.accuracy(x_batch, t_batch)
				test_acc = network.accuracy(x_test, t_test)
				network.train_acc_list.append(train_acc)
				network.test_acc_list.append(test_acc)
				print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


		return network

if __name__ == '__main__':
	
	# 演示训练一个模型
	start = time.perf_counter()
	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
	network = training(0.1, 100, 0.0001, x_train, t_train, x_test, t_test, 100000, 100)
	torch.save(network, 'best_model.pth')

	# 画损失函数的变化
	x1 = np.arange(len(network.test_loss_list))
	ax1 = plt.subplot(121)
	plt.plot(x1, network.train_loss_list, label='train loss')
	plt.plot(x1, network.test_loss_list, label='test loss', linestyle='--')
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.legend(loc='upper right')



	# 画训练精度，测试精度随着epoch的变化
	markers = {'train': 'o', 'test': 's'}
	x2 = np.arange(len(network.train_acc_list))
	ax2 = plt.subplot(122)
	plt.plot(x2, network.train_acc_list, label='train acc')
	plt.plot(x2, network.test_acc_list, label='test acc', linestyle='--')
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.ylim(0, 1.0)
	plt.legend(loc='lower right')
	plt.show()



	best_model = network
	w_1 = np.array(best_model.params['W1'])
	w_2 = np.array(best_model.params['W2'])
	p = int(np.sqrt(w_2.shape[0]))
	# 进行测试
	ac = best_model.accuracy(x_test, t_test)
	print(f"分类精度为: {ac}")

	# 可视化网络参数

	# PCA 降维
	pca=PCA(n_components=3)
	w_1_pca = pca.fit_transform(w_1).reshape(28,28,3)

	#归一化
	w_1_pca = w_1_pca / w_1_pca.max() 
	w_1_pca = 255 * w_1_pca
	w_1_pca = w_1_pca.astype(np.uint8)
	imageio.imwrite('W1.png', w_1_pca)

	for i in range(10):
	    a = np.array(w_2[:,i]).reshape(p,p)
	    a = a / a.max()
	    a = 255 * a
	    a = a.astype(np.uint8)
	    imageio.imwrite(f"{i}_w_2.png", a)

	end = time.perf_counter()
	print('Running Time: %s Seconds' %(end-start))