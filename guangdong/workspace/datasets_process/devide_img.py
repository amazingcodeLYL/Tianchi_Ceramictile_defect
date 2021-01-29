import os
from tqdm import tqdm
import os.path as osp
import cv2
import random
import json

def devide_img(img_path,save_path):
    '''
    将大图切割成小图
    :param img_path: 每一个未切割的路径
    :param save_path: 切割后存储路径
    :return:
    '''
    img_list=os.listdir(img_path)
    for img in tqdm(img_list):
        for j in range(100):
            fname,ext=osp.splitext(img)
            img_path_name=img_path+'/'+img
            img_numpy=cv2.imread(img_path_name)
            w=random.randint(0,img_numpy.shape[0]-128)
            h=random.randint(0,img_numpy.shape[1]-128)
            new_img=img_numpy[w:w+128,h:h+128]
            new_img_name=fname+'_'+str(j)+ext
            new_img_path=osp.join(save_path,new_img_name)
            cv2.imwrite(new_img_path,new_img)

def get_roi(img_path,json_file_path):
    '''
    提取图片里标注的roi区域
    :param img_path:
    :param json_file_path:
    :return:
    '''
    img_list = os.listdir(img_path)
    with open(json_file_path) as f:
        json_file=json.load(f)
        cnt=0
        for i_img in img_list:
            for line in json_file:
                if line['name']==i_img:
                    bbox=line['bbox']
                    img = cv2.imread(img_path + i_img)
                    img=img[int(bbox[0]):int(bbox[2]),int(bbox[1]):int(bbox[3])]
                    save_roi_path=os.path.join(img_path,i_img)
                    print(save_roi_path)
                    cv2.imwrite(save_roi_path, img)
                    cv2.imshow('img',img)
                    cv2.waitKey(0)
                    cv2.destroyWindow('img')




if __name__=="__main__":
    img_path="/home/dell/桌面/img/"
    # save_path="/home/dell/桌面/devide_img/"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # devide_img(img_path,save_path)

    # json_file_path="/home/dell/桌面/tile_round1_train_20201231/train_annos.json"
    # get_roi(img_path,json_file_path)

    from PIL import Image
    import numpy
    img=Image.open(img_path+"erha.jpg")
    icon=Image.open(img_path+"197_25_t202011190901565_CAM2.jpg")
    img = img.convert("RGBA")

    icon.resize((5,5),Image.ANTIALIAS)
    img.paste(icon,(200,200),mask=None)
    img.show()


