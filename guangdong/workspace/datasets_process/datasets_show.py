import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
data_root='/home/dell/桌面/tile_round1_train_20201231/train_imgs/'
json_file="/home/dell/桌面/tile_round1_train_20201231/train_annos.json"
# with open(json_file) as f:
#     json_file=json.load(f)
#     for i in range(100):
#         name = json_file[i]['name']
#         # print(data_root+name)
#         h = json_file[i]['image_height']
#         w = json_file[i]['image_width']
#         # 缩放图片
#         scale = max(h, w) / float(1000)
#         print(scale)
#         new_w, new_h = int(w / scale), int(h / scale)
#         img = cv2.imread(data_root + name)
#         img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#         # cv2.imshow('img',img)
#         # cv2.waitKey()
#         # cv2.destroyWindow()
#
#         category = json_file[i]['category']
#         bbox = json_file[i]['bbox']
#         print(bbox)
#         x1 = int(bbox[0] / scale)
#         y1 = int(bbox[1] / scale)
#         x2 = int(bbox[2] / scale)
#         y2 = int(bbox[3] / scale)
#         print(x1, y1, x2, y2)
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
#         # plt.imshow(img)
#         cv2.imshow('image', img)
#         cv2.waitKey()
#         cv2.destroyWindow('image')


"""
统计数据集中框的长宽比
"""
import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)

# 标注长宽高比例
box_w = []
box_h = []
box_wh = []
# 读取数据
ann_json = '/home/dell/桌面/test_code_imgs/train_annos.json'
with open(ann_json) as f:
    annotation=json.load(f)
    for ann in annotation:
        box_w.append(round(ann['bbox'][2], 2))
        box_h.append(round(ann['bbox'][3], 2))
        wh = round(ann['bbox'][2] / ann['bbox'][3], 0)
        if wh < 1:
            wh = round(ann['bbox'][3] / ann['bbox'][2], 0)
        box_wh.append(wh)

box_wh_unique = list(set(box_wh))
box_wh_count=[box_wh.count(i) for i in box_wh_unique]

# 绘图
wh_df = pd.DataFrame(box_wh_count,index=box_wh_unique,columns=['宽高比数量'])
wh_df.plot(kind='bar',color="#55aacc")
plt.show()

#################################################################################################
#创建类别标签字典
# category_dic=dict([(i['id'],i['name']) for i in ann['categories']])
# counts_label=dict([(i['name'],0) for i in ann['categories']])
# for i in ann['annotations']:
#     counts_label[category_dic[i['category_id']]]+=1

# # 标注长宽高比例
# box_w = []
# box_h = []
# box_wh = []
# categorys_wh = [[] for j in range(10)]
# for a in ann['annotations']:
#     if a['category_id'] != 0:
#         box_w.append(round(a['bbox'][2],2))
#         box_h.append(round(a['bbox'][3],2))
#         wh = round(a['bbox'][2]/a['bbox'][3],0)
#         if wh <1 :
#             wh = round(a['bbox'][3]/a['bbox'][2],0)
#         box_wh.append(wh)
#
#         categorys_wh[a['category_id']-1].append(wh)
#
#
# # 所有标签的长宽高比例
# box_wh_unique = list(set(box_wh))
# box_wh_count=[box_wh.count(i) for i in box_wh_unique]
#
# # 绘图
# wh_df = pd.DataFrame(box_wh_count,index=box_wh_unique,columns=['宽高比数量'])
# wh_df.plot(kind='bar',color="#55aacc")
# plt.show()