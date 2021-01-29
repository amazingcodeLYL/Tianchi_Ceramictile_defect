import cv2
import json
import os.path as osp
import os

img_path="/home/dell/桌面/tile_round1_testA_20201231/tile_round1_testA_20201231/testA_imgs"
save_img_path=osp.dirname(img_path)
save_img_path=osp.join(save_img_path,'json_img')
if not osp.exists(save_img_path):
    os.makedirs(save_img_path)
with open("/home/dell/桌面/guangdong/guangdong/final_result.json") as fp:
    json_file=json.load(fp)
    for line in json_file:
        img_name=line['name']
        img_p=osp.join(img_path,img_name)
        bbox=line['bbox']
        x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
        img=cv2.imread(img_p)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        save_path=osp.join(save_img_path,img_name)
        print(save_path)
        cv2.imwrite(save_img_path,img)
        print("save##########")
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyWindow('img')

