# ----------------------------------------------------------------------#
#   验证集的划分在train.py代码里面进行
#   test.txt和val.txt里面没有内容是正常的。训练不会使用到。
# ----------------------------------------------------------------------#
'''
#--------------------------------注意----------------------------------#
如果在pycharm中运行时提示：
FileNotFoundError: [WinError 3] 系统找不到指定的路径。: './VOCdevkit/VOC2007/Annotations'
这是pycharm运行目录的问题，最简单的方法是将该文件复制到根目录后运行。
可以查询一下相对目录和根目录的概念。在VSCODE中没有这个问题。
#--------------------------------注意----------------------------------#
'''
import os
import random

random.seed(0)

txtfilepath = '../voc_hole/TXT/'
saveBasePath = '../voc_hole/'

# ----------------------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   train_percent不需要修改
# ----------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9

temp_txt = os.listdir(txtfilepath)
total_txt = []
for txt in temp_txt:
    if txt.endswith(".txt"):
        total_txt.append("myNKL_resize_100_100/" + txt)

num = len(total_txt)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train size", tr)
ftrainval = open(os.path.join(saveBasePath, 'my_nkl_trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'my_nkl_test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'my_nkl_train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'my_nkl_val.txt'), 'w')

for i in list:
    name = total_txt[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
