from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2
import math
import matplotlib.pyplot as plt
# from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
import torch.nn.functional as F
from models.model import create_model, load_model
from datasets.dataset.jde import JointDataset, collate_fn
from models.utils import _tranpose_and_gather_feat
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from sklearn import manifold
import seaborn as sns


def collect_emb(
        opt,
        batch_size=1,
        img_size=(1088, 608),
        print_interval=40,
):
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    nC = 1
    test_paths = data_cfg_dict['test_emb']
    dataset_root = data_cfg_dict['root']
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, '/media/ubuntu/e1c57dab-f3ee-4ae3-839f-38a0448c56df/ms1/FairMOT-master/models/ctdet_coco_dla_2x.pth')
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(opt, dataset_root, test_paths, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=8, drop_last=False)
    embedding, id_labels = [], []
    feature_collector = {}
    print('Collecting pedestrain features...')
    for batch_i, batch in enumerate(dataloader):
        t = time.time()
        output = model.f_heads(model.forward_tr(batch['input'].cuda()))[-1]
        # 获取每一帧的id_head
        id_head = _tranpose_and_gather_feat(output['id'], batch['ind'].cuda())
        id_head = id_head[batch['reg_mask'].cuda() > 0].contiguous()
        emb_scale = math.sqrt(2) * math.log(opt.nID - 1)
        id_head = emb_scale * F.normalize(id_head)
        id_target = batch['ids'].cuda()[batch['reg_mask'].cuda() > 0]

        id_head = id_head.detach().cpu().numpy()
        id_target = id_target.detach().cpu().numpy()

        for i in range(0, id_head.shape[0]):
            if len(id_head.shape) == 0:
                continue
            elif id_target[i] not in feature_collector.keys():
                id_emb = id_head[i][np.newaxis, :]
                feature_collector.update({id_target[i]: id_emb})

            else:
                id_emb = id_head[i][np.newaxis, :]
                feature_collector[id_target[i]] = np.append(feature_collector[id_target[i]], id_emb, axis=0)

        if batch_i == 15:
            return feature_collector


def visual(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=23)
    X_tsne = tsne.fit_transform(X)
    print(X.shape, X_tsne.shape)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    x_final = (X_tsne - x_min) / (x_max - x_min)
    return x_final


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    with torch.no_grad():
        tpr = collect_emb(opt, batch_size=1)
    print(tpr.keys())

    plt.figure(figsize=(8, 8))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(tpr)))
    print(colors[0])
    M_pool = np.empty(shape=(0, 128))
    for key, value in tpr.items():
        M_pool = np.append(M_pool, values=value, axis=0)
    x_final = visual(M_pool)
    m_start = 0
    j = 0
    for key, value in tpr.items():
        num = len(value)
        m = x_final[m_start: m_start + num - 1]
        for i in range(len(m)):
            plt.text(m[i, 0], m[i, 1], '*', color=colors[j], fontdict={'weight': 'bold', 'size': 20})

        j += 1
        m_start = m_start + num
    plt.xticks([])
    plt.yticks([])
    plt.savefig('/media/ubuntu/e1c57dab-f3ee-4ae3-839f-38a0448c56df/ms1/FairMOT-master/dataset/MOT17/images/matching/1tsne.png')
    #plt.show()
