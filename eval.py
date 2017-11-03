import matplotlib.pyplot as plt
import numpy as np

log_path = './result/log_CNN_2'
f = open(log_path)
lines = f.readlines()
f.close()

x = np.arange(1, 101, 1);
train_l = np.zeros(100)
train_a = np.zeros(100)
test_l = np.zeros(100)
test_a = np.zeros(100)
for i in range(len(lines)):
    if i % 2 == 1:
        continue
    line = lines[i]
    d = line.split(' ')
    # print d
    train_l[i/2] = float(d[3])
    train_a[i/2] = float(d[5])
    test_l[i/2] = float(d[7])
    test_a[i/2] = float(d[9])

plt.figure(1)
plt.plot(x, train_a, 'b', linewidth=2, label="Train Accuracy")
plt.plot(x, test_a, 'r', linewidth=2, label='Test Accuracy')
plt.legend(loc="lower right")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.figure(2)
plt.plot(x, train_l, 'b', linewidth=2, label="Train Loss")
plt.plot(x, test_l, 'r', linewidth=2, label='Test Loss')
plt.legend(loc="upper right")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

