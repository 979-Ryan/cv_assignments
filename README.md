# cv_assignments
1. 文件放置位置说明：需要将本人上传的所有文件下载并放置在同一个目录下，维持当前的并列与包含关系即可。
   特别说明：定义load_mnist()方法的mnist.py文件，放在dataset文件夹中。
2. 已将一个训练好的模型“best_model.pth”一并上传，不进行训练，即可直接测试，只需直接运行test.py文件，就能调用“best.model.pth”对测试集进行测试，并输出测试精度  
3. train.py文件中定义了神经网络类twoNN和训练函数training,直接运行train.py文件，将自动训练一个模型，超参数已经被简单设定为学习率：0.1，隐藏层大小：100，
正则化强度0.0001，迭代次数：100000，batch_size：100。
