
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import seaborn as sns
import h5py

def visual(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print(X.shape, X_tsne.shape)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    x_final = (X_tsne - x_min) / (x_max - x_min)
    return x_final

h5f_feat = h5py.File(
                '/media/ubuntu/e1c57dab-f3ee-4ae3-839f-38a0448c56df/ms1/FairMOT-master/dataset/MOT17/feat.h5', 'r')
h5f_id = h5py.File(
                '/media/ubuntu/e1c57dab-f3ee-4ae3-839f-38a0448c56df/ms1/FairMOT-master/dataset/MOT17/id.h5', 'r')
file_index=[]
for line in open("/media/ubuntu/e1c57dab-f3ee-4ae3-839f-38a0448c56df/ms1/FairMOT-master/dataset/MOT17/file_index.txt","r"):
    file.append(line)
feat = h5f_feat[file_index[0]][:]
id = h5f_id[file_index[0]][:]
for i in range(1, len(file_index)):
    feat = torch.cat([feat, h5f_feat[file_index[i]][:]], dim=0)
    id = torch.cat([id, h5f_id[file_index[i]][:]], dim=0)


id_n = {}
num=list(set(id))

for i in range(len(num)):
    id_n[num[i]] = i
x_final = visual(feat)
plt.figure(figsize=(8,8))
for i in range(x_final.shape[0]):
    plt.scatter(x_final[i,0], x_final[i,1], color=plt.cm.Set1(id_n[id[i]]), cmap='winter_r')
    #plt.text(x_final[i,0], x_final[i,1], str(id_n[y[i]]), color=plt.cm.Set1(id_n[y[i]]), fontdict={'weight':'bold', 'size':9})
plt.xticks([])
plt.yticks([])
plt.show()