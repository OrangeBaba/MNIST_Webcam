# 学習と推論
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer
import time

# 時間計測開始
time_sta = time.time()

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
max_epochs = 20
# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:10000], t_train[:10000]
#x_test, t_test = x_test[:2000], t_test[:2000]

# 学習情報出力
train_num = "train" + str(x_train.shape[0])
test_num = "test" + str(x_test.shape[0])
epochs_num = "epoch" + str(max_epochs)
print(train_num)
print(test_num)
print(epochs_num)

# 学習処理
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
save_name = (train_num + "_" + test_num + "_" + epochs_num + '.pkl')
print(save_name)
network.save_params(save_name)
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 時間計測終了
time_end = time.time()
# 経過時間（秒）
tim = time_end- time_sta
print(tim, "sec")
