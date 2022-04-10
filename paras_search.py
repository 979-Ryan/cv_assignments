"""超参数的随机搜索"""
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from train import training

def shuffle_dataset(x,t):
	# 打乱数据集
	p = np.random.permutation(x.shape[0])
	x = x[p]
	t = t[p]
	return x,t

if __name__ == '__main__':
	
	start = time.perf_counter()

	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

	# 选取前20%作为验证集
	validation_rate = 0.2
	validation_num = int(x_train.shape[0]*validation_rate)
	x_train, t_train = shuffle_dataset(x_train, t_train)
	x_val = x_train[:validation_num]
	t_val = t_train[:validation_num]
	x_train = x_train[validation_num:]
	t_train = t_train[validation_num:]

	results_val = {}
	results_train = {}
	hsizes = [100, ]
	ite = 12

	models = {}
	for i in range(ite):
		learning_rate = np.around(np.random.uniform(0.09,0.11),3)
		lam= np.around(np.random.uniform(0.00009,0.00011),5)
		hidden_size = np.random.choice(hsizes)

		model = training(learning_rate, hidden_size, lam, x_train, t_train, x_val, t_val, iters_num=20000, batch_size=100)
		val_acc_list = model.test_acc_list
		print('验证精度'+str(val_acc_list[-1])+'|学习率:'+str(learning_rate)+ \
				'权值衰减:'+str(lam)+'隐藏层大小:'+str(hidden_size))
		key = 'learning_rate:'+str(learning_rate)+',lambda:'+str(lam)+'hidden_size:'+str(hidden_size)
		models[key] = model
		results_val[key] = val_acc_list
		results_train[key] = model.train_acc_list

	# 画图
	print('----最优超参数降序排列的结果----')
	i = 0
	for key, val_acc_list in sorted(results_val.items(), key = lambda x : x[1][-1], reverse = True):
		print('最优验证精度:'+str(val_acc_list[-1])+'|'+key)
		if i == 0:
			torch.save(models[key], 'best_model.pth')
		markers = {'train': 'o', 'val': 's'}
		plt.subplot(3,4,i+1)
		plt.title('Best-'+str(i+1))
		plt.ylim(0.0,1.0)
		if i % 4:
			plt.yticks([])
		plt.xticks([])
		x = np.arange(len(val_acc_list))
		plt.plot(x, val_acc_list,label='val acc')
		plt.plot(x, results_train[key],label='train acc',linestyle = '--')
		plt.xlabel("epochs")
		plt.ylabel("accuracy")
		plt.legend(loc='lower right')
		i += 1
		if i >= 12:
			break
	plt.show()

	end = time.perf_counter()
	print('Running Time: %s Seconds' %(end-start))


