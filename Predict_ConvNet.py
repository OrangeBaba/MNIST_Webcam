# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params("パラメータのパス")

print("calculating test accuracy ... ")

# 黒背景に白文字でないとうまく判断できない。（白黒画像なので。。。）
img = Image.open('画像のパス').convert('L')
plt.imshow(img)

img.thumbnail((28, 28)) # 28*28に変換
img = np.array(img).reshape(1,28,28) # numpy arrayに変換
print(img.shape)

pred = network.predict(img[np.newaxis])

print("======= misclassified result =======")
print("{view index: (label, inference), ...}")

print(np.argmax(pred))
print(pred)