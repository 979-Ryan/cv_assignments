# cv_assignments
1. 文件放置位置说明：需要将本人上传的所有文件下载并放置在同一个目录下，维持当前的并列与包含关系即可。
   特别说明：定义load_mnist()方法的mnist.py文件，放在dataset文件夹中。
   
2. 关于模型的训练：
   train.py定义了神经网络类twoNN以及模型训练方法training，直接运行train.py文件，可以训练出一个模型并存储在“best_model.pth”文件中，此模型的超参数是给定的，可以根据需要改动。
   模型训练过程中的accuracy变化将会被输出，并且将绘制loss与accuracy曲线，同时可视化网络参数并保存至当前目录中。
   
3. 关于参数查找：
   paras_search.py文件根据对超参数的范围的设定进行随机搜索，并且调用了train.py中的training方法。直接运行paras_search.py文件可以进行模型训练与超参数查找，程序会根据在
   验证集上的accuracy对模型进行排序，并输出相应的准确率与超参数，绘制前12名模型的accuracy曲线。效果最好的模型会被保存在“best_model.pth”文件中，同样，此模型的超参数范围与寻找
   次数也是可以改动的。
   
4. 关于模型的测试：
   直接运行test.py文件，就可以用保存在“best_model.pth”文件中的模型对测试集进行测试，并输出测试精度。注意：“best_model.pth”文件是可覆写的，目前文件中已经存储了本人训练的最优模型。
 
