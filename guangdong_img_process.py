""""
author:lyl
date:1/14/2021
"""
import  cv2
import numpy as np
import  os.path as osp
import os
import shutil
import random
import json

def img_devided_to_cam1_cam2_cam3(img_path):
    """
    将train_imgs分成三类文件夹，cam1、cam2、cam3
    """
    dirname=osp.dirname(img_path)
    cam1=osp.join(dirname,'cam1','train_imgs')
    cam2=osp.join(dirname,'cam2','train_imgs')
    cam3=osp.join(dirname,'cam3','train_imgs')
    if not osp.exists(cam1):
        os.makedirs(cam1)
    if not osp.exists(cam2):
        os.makedirs(cam2)
    if not osp.exists(cam3):
        os.makedirs(cam3)

    img_list=os.listdir(img_path)
    for img in img_list:
        path=osp.join(img_path,img)
        if 'CAM1' in img:
            cam1_path=osp.join(cam1,img)
            shutil.copyfile(path,cam1_path)
        elif 'CAM2' in img:
            cam2_path=osp.join(cam2,img)
            shutil.copyfile(path,cam2_path)
        elif 'CAM3' in img:
            cam3_path=osp.join(cam3,img)
            shutil.copyfile(path,cam3_path)

def get_roi_img(img_path,json_path):
    """
    提取出大图中的缺陷区域，并将它保存
    """
    img_list=os.listdir(img_path)
    with  open(json_path) as f:
        import json
        json_file=json.load(f)
    dirname=osp.dirname(img_path)
    save_path=osp.join(dirname,'croped_roi/')
    if not osp.exists(save_path):
        os.makedirs(save_path)
    for img in img_list:
        path=osp.join(img_path,img)
        for json in json_file:
            if img==json['name']:
                bbox_=json['bbox']
                category_=json['category']
                image=cv2.imread(path)
                roi_image=image[int(bbox_[0]):int(bbox_[2]),int(bbox_[1]):int(bbox_[3])]
                w=bbox_[2]-bbox_[0]
                h=bbox_[3]-bbox_[1]
                if roi_image.size !=  0:
                    cv2.imwrite(save_path+"croped_roi_img{}_{}_{}.jpg".format(int(w),int(h),category_), roi_image)
                # cv2.imshow('img',image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows('img')

def aug_normal(defect_img_path,normal_img_path,transform):
    """
    正常的数据增广，将瑕疵块融合进原图大图中,并将其写入json中，并将融合后的图片保存下来
    """
    img_path=osp.dirname(normal_img_path)
    save_augimg_path=osp.join(img_path,'aug_img/')
    print(save_augimg_path)
    if not osp.exists(save_augimg_path):
        os.makedirs(save_augimg_path)
    defect_img_list=os.listdir(defect_img_path)
    normal_img_list=os.listdir(normal_img_path)
    random.seed(2021)
    result=[]
    for img in normal_img_list:
        img_path=osp.join(normal_img_path,img)
        normal_image=cv2.imread(img_path)
        # height,width=image.shape[0],image.shape[1]
        for j in range(15):
            defect_name = random.choice(defect_img_list)
            defect_path=osp.join(defect_img_path,defect_name)
            defect_image = cv2.imread(defect_path)
            defect_image = transform(image=defect_image)['image']
            label = defect_name.split('.jpg')[0].split('_')[-1]
            defect_h, defect_w = defect_image.shape[0], defect_image.shape[1]
            # print(label)
            # print(defect_h, defect_w)
            xmin=random.randint(1,8192)
            ymin=random.randint(1,6000)
            xmax=xmin+defect_w
            ymax=ymin+defect_h
            mask = 255 * np.ones((defect_h + 2, defect_w + 2, 3), dtype=np.uint8)
            mask[0, :] = 0
            mask[defect_h, :] = 0
            mask[:, 0] = 0
            mask[:, defect_w] = 0
            # import pdb
            # pdb.set_trace()
            try:
                image = cv2.seamlessClone(defect_image, normal_image, mask, (xmin + defect_h // 2,ymin + defect_w // 2), cv2.MIXED_CLONE)
                image=cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                # cv2.imshow('img', image)
                # cv2.waitKey(0)
                # cv2.destroyWindow('img')

                #     # normal_image[ymin:ymax, xmin:xmax] = defect_image
                #     # result.append({'name': img, 'category': label, 'bbox': [xmin, ymin, xmax, ymax]})
                aug_img_path = osp.join(save_augimg_path, img)
                cv2.imwrite(aug_img_path, image)
                normal_image = image
            except:
                continue

            print("save##########")
    json_name="/home/dell/桌面/Tianchi_Fabric_defects_detection/tools/data_process/aug_json.json"
    with open(json_name,'w') as fp:
        json.dump(result,fp,indent=4,separators=(',',':'))






if __name__=="__main__":
    # img_path="/home/dell/桌面/test_code_imgs/imgs"
    # json_path="/home/dell/桌面/test_code_imgs/train_annos.json"
    # img_devided_to_cam1_cam2_cam3("/home/lyl/桌面/train_images/train_imgs")
    # get_roi_img(img_path,json_path)
    img_path="/home/dell/桌面/tile_round1_train_20201231/train_imgs/"
    # defect_img_path="/home/dell/桌面/test_code_imgs/croped_roi/"
    # import albumentations as A
    #
    # transform_3456 = A.Compose([  # 对第3,4,5,6类缺陷做增广
    #     A.RandomScale(),
    #     A.Flip(),
    #     A.GridDistortion(distort_limit=0.3, num_steps=5),
    #     A.RandomRotate90()
    # ])
    # aug_normal(defect_img_path,img_path,transform_3456)
    img_devided_to_cam1_cam2_cam3(img_path)