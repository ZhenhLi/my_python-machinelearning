import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)



def plot_decision_boundary(model,axis):  # 两个数据特征基础下输出决策边界函数
    x0,x1=np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1,1)
    )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_pre=model.predict(x_new)
    zz=y_pre.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    cus=ListedColormap(["#EF9A9A","#FFF59D","#90CAF9"])
    plt.contourf(x0,x1,zz,cmap=cus)

d = datasets.load_iris()

x = d.data[:,:2]
y = d.target.astype(np.float64)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=555)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
svm_clf = Pipeline(( ("scalar", StandardScaler()),("linear_svc", LinearSVC(C=1, loss="hinge")) ,))
svm_clf.fit(x_train, y_train)

print(svm_clf.score(x_test, y_test))
plt.figure(1)
# plt.figure("svm_clf")
# ax1 = plt.subplot(1,3,1)
# plt.sca(ax1)
plot_decision_boundary(svm_clf, axis=[4,9,1,5])

plt.scatter(x[y==0,0],x[y==0,1],color="r")
plt.scatter(x[y==1,0],x[y==1,1],color="g")
plt.scatter(x[y==2,0],x[y==2,1],color="b")
plt.savefig("test1_svm.jpg")
# plt.imshow(ax1)

# OVR, 即不输入参数时
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print(log_reg.score(x_test, y_test))
# plt.figure("test2_lr_OVR")
# ax2 = plt.subplot(1,3,2)
# plt.sca(ax2)
plot_decision_boundary(log_reg, axis=[4,9,1,5])

plt.scatter(x[y==0,0],x[y==0,1],color="r")
plt.scatter(x[y==1,0],x[y==1,1],color="g")
plt.scatter(x[y==2,0],x[y==2,1],color="b")
plt.savefig("test2_lr_OVR.jpg")

# plt.imshow(ax2)


# OVO,
log_reg1 = LogisticRegression(multi_class="multinomial",solver="newton-cg")
log_reg1.fit(x_train, y_train)
print(log_reg1.score(x_test, y_test))
# plt.figure("test3_OVO_lr")
# ax3 = plt.subplot(1,3,3)
# plt.sca(ax3)
plot_decision_boundary(log_reg1, axis=[4,9,1,5])

plt.scatter(x[y==0,0],x[y==0,1],color="r")
plt.scatter(x[y==1,0],x[y==1,1],color="g")
plt.scatter(x[y==2,0],x[y==2,1],color="b")
plt.savefig("test3_OVO_lr.jpg")

# plt.imshow(ax3)
plt.show()
# plt.savefig("contrast_of_svm_lrOVR_lrOVO.jpg")

x1 = d.data
y1 = d.target

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,random_state=555)

log_reg11 = LogisticRegression()
log_reg11.fit(x1_train,y1_train)
print(log_reg11.score(x1_test,y1_test))

log_reg12=LogisticRegression(multi_class="multinomial",solver="newton-cg")
log_reg12.fit(x1_train,y1_train)
print(log_reg12.score(x1_test,y1_test))