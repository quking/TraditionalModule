import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
这个函数是用来画边界图，需要传入一个模型进去，并且指定相应的范围 ，这个类主要是看分类效果
下面给出一个例子：
log_reg = LogisticRegression()  # 引入逻辑回归模型
log_reg.fit(X, y)    # 用整个数据集进行训练

plot_decision_boundary(log_reg, axis = [-4,4,-4,4])   # 指定x轴的范围和y轴的范围
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()
"""


class DecisonPlot:

    def plot_decision_boundary(model, axis):
        x0, x1 = np.meshgrid(
            np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
            np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
        )
        X_new = np.c_[x0.ravel(), x1.ravel()]
        y_predict = model.predict(X_new)
        zz = y_predict.reshape(x0.shape)
        custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
        plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

