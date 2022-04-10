import torch
import numpy as np
from train import twoNN
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import imageio
from sklearn.decomposition import PCA 

if __name__ == '__main__':
    
    # 导入测试集与模型
    (_,_), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    best_model = torch.load('best_model.pth')

    # 进行测试
    ac = best_model.accuracy(x_test, t_test)
    print(f"分类精度为: {ac}")

