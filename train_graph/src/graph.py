import matplotlib.pyplot as plt
import numpy as np
'''
x=[2,4,8,16,32,64]
y=[0.7864,0.8668,0.8698,0.8689,0.8568,0.8668]
x_values = np.linspace(0, 7, num=6)
plt.plot(x,y,'o-',color='b')
x=[16.7,33.4,50.1,66.664,83.33,100]
plt.xticks(x,(["2","4","8","16","32","64"]))
plt.xlabel("dimension")
plt.ylabel("F1_score")
plt.savefig("/home/c402/LhL/RippleNet/RippleNet/aaa.jpg",dpi=600)
'''
'''
x=[1,2,3,4,5]
# y1=[0.8458,0.8989,0.9467,0.9640,0.8942]
# y2=[0.8815,0.8962,0.9224,0.9445,0.8776]
# y1=[0.0163,0.0028,0.1250,0.0329,0.0665]
# y2=[0.0703,0.0208,0.1901,0.0713,0.1901]
y1=[0.7567,0.5092,0.9030,0.9639,0.9269]
y2=[0.7731,0.5123,0.9107,0.9454,0.9027]
# x_values = np.linspace(0, 7, num=6)
plt.plot(x,y1,'o-',color= 'steelblue',label="Cert-r4.2")
plt.plot(x,y2,'o-',color='forestgreen',label="Cert-r5.2")
# x=[16.7,33.4,50.1,66.664,83.33,100]
plt.xticks(x,(["0.05","0.1","0.5","1","1.5"]))
plt.xlabel("T")
plt.ylabel("F1_Score")
plt.legend(loc='upper right')
plt.savefig("/home/c402/LhL/RippleNet/RippleNet/src/F1_r.jpg",dpi=600)
'''

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
'''
X=np.array([[0,3,2,4],[5,4,7,8],[9,16,8,5],[13,3,4,16],[6,18,1,20]])
X=np.array([[0.9500,0.3519,0.5135],
            [0.5018,0.9410,0.6546],
            [0.7259,0.9026,0.8047],
            [0.9300,0.9187,0.9243],
            [0.9433,0.9328,0.9380]])
A = np.arange(0, 100).reshape(10, 10)
plt.figure(figsize=(6,8))
ax = plt.matshow(X, cmap='GnBu',fignum=0)
plt.colorbar(ax.colorbar, fraction=0.03)
plt.tick_params(axis='x', labelbottom=True, labeltop=False)
plt.yticks(np.arange(5),["Iforest","AutoEncoder","Deeplog","Idtbert","Ours"],fontsize=14)
plt.xticks(np.arange(3),(["precision","recall","F1"]),fontsize=14)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        plt.text(j,i,f'{X[i,j]:.3f}',ha='center',va='center',color='black',fontsize=14)
plt.savefig("/home/c402/LhL/RippleNet/RippleNet/src/cons.jpg",dpi=600)
'''
'''
data1=[0.9967,0.9955,0.9959,0.9988]
data2=[0.9989,0.9977,0.9968,0.9864]
data3=[0.3350,0.2321,0.2697,0.2500]
data4=[0.8045,0.8726,0.8695,0.8699]
labels = ['attack1', 'attack2', 'attack3', 'attack4']
width1=range(0,len(data1))
width1=[i+0.2 for i in width1]
width2=[i+0.2 for i in width1]
width3=[i+0.4 for i in width1]
width4=[i+0.6 for i in width1]
plt.bar(width1, data1, tick_label=labels,lw=0.5,width=0.2,fc="sandybrown",label="plus_transform")
plt.bar(width2, data2, tick_label=labels,lw=0.5,width=0.2,fc="silver",label="plus")
plt.bar(width3, data3, tick_label=labels,lw=0.5,width=0.2,fc="lightskyblue",label="replace")
plt.bar(width4, data4, tick_label=labels,lw=0.5,width=0.2,fc="thistle",label="replace_transform")
plt.ylabel("Accuracy")
plt.xlabel("Attack Type")
plt.legend(loc='lower left')
plt.savefig("/home/c402/LhL/RippleNet/RippleNet/src/bar.jpg",dpi=600)
'''
'''
data1=[0.9903,0.9829,0.9850,0.9963]
data2=[0.9967,0.9914,0.9883,0.9581]
data3=[0.3350,0.2321,0.2397,0.2500]
data4=[0.9090,0.8936,0.8941,0.8964]
'''
data1=[0.9951,0.9914,0.9924,0.9982]
data2=[0.9984,0.9957,0.9941,0.9744]
data3=[0.5019,0.4153,0.4249,0.4000]
data4=[0.6133,0.7059,0.7076,0.7083]
labels = ['attack1', 'attack2', 'attack3', 'attack4']
width1=range(0,len(data1))
width1=[i+0.2 for i in width1]
width2=[i+0.2 for i in width1]
width3=[i+0.4 for i in width1]
width4=[i+0.6 for i in width1]
plt.bar(width1, data1, tick_label=labels,lw=0.5,width=0.2,fc="sandybrown",label="plus_transform")
plt.bar(width2, data2, tick_label=labels,lw=0.5,width=0.2,fc="silver",label="plus")
plt.bar(width3, data3, tick_label=labels,lw=0.5,width=0.2,fc="lightskyblue",label="replace")
plt.bar(width4, data4, tick_label=labels,lw=0.5,width=0.2,fc="thistle",label="replace_transform")
plt.ylabel("F1")
plt.xlabel("Attack Type")
plt.legend(loc='lower left')
plt.savefig("/home/c402/LhL/RippleNet/RippleNet/src/F1_bar.jpg",dpi=600)
