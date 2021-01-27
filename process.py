import os
import os.path as osp
import json
import cv2
import numpy as np
import argparse
import shutil
from tqdm import tqdm
import albumentations as A
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def angle_detection( image):
    """
    角点检测
    :param image:
    :return:
    """
    image = cv2.resize(image,(800,700))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 3, 3, 0.01)
    image[dst > 0.7 * dst.max()] = [0, 0, 255]
    # cv2.namedWindow('img',flags=0)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyWindow('img')


def test_edge():
    pass

class ProcessData:
    def __init__(self,json_file,test_path,train_path):
        self.json_file = json_file
        self.test_path = test_path
        self.train_path = train_path

    def load_json(self,json_file):
        '''
        将coco格式的标注读取出来，并且转化为
        {
        image_name:[bbox,bbox....]
        }
        的格式
        :return:转化好的字典以及另外一个file_name到id,width,height的字典
        '''
        # import pdb
        # pdb.set_trace()
        coco = json.load(open(json_file))
        images = coco['images']
        annotations = coco['annotations']
        im2ann_dict = {}
        im_id2im_name_dict = {}
        im_name2im_id_dict = {}
        for im in images:
            im_id2im_name_dict[im['id']] = im['file_name']
            im_name2im_id_dict[im['file_name']] = [im['id'],im['width'],im['height']]
        for ann in annotations:
            if im_id2im_name_dict[ann['image_id']] not in im2ann_dict.keys():
                im2ann_dict[im_id2im_name_dict[ann['image_id']]] = []
            im2ann_dict[im_id2im_name_dict[ann['image_id']]].append([ann['category_id'], ann['bbox']])
        return im2ann_dict,im_name2im_id_dict

    def devide_image(self,img_path):
        '''
        将原本的图片按照不同的相机分成三类
        :param img_path:
        :return:
        '''
        father_fold = osp.dirname(img_path)
        cam1 = osp.join(father_fold,'cam1','train_img')
        cam2 = osp.join(father_fold,'cam2','train_img')
        cam3 = osp.join(father_fold,'cam3','train_img')
        if not osp.exists(cam1):
            os.makedirs(cam1)
        if not osp.exists(cam2):
            os.makedirs(cam2)
        if not osp.exists(cam3):
            os.makedirs(cam3)
        imgs = os.listdir(img_path)
        for img in tqdm(imgs):
            path = osp.join(img_path,img)
            if 'CAM1' in path:
                cam1_path = osp.join(cam1,img)
                shutil.copyfile(path,cam1_path)
            elif 'CAM2' in path:
                cam2_path = osp.join(cam2,img)
                shutil.copyfile(path,cam2_path)
            elif 'CAM3' in path:
                cam3_path = osp.join(cam3,img)
                shutil.copyfile(path,cam3_path)

    def crop_image_train(self,img_path,annotations,crop_h,crop_w,json_name,im_fold):
        # import pdb
        # pdb.set_trace()
        save_path = osp.dirname(img_path)
        json_path = osp.join(save_path, json_name)
        save_path = osp.join(save_path, im_fold)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        imgs = os.listdir(img_path)
        # crop = A.Compose(
        #     [A.RandomCrop(crop_h, crop_w)],
        #     bbox_params=A.BboxParams(format='coco')
        # )
        # image = cv2.imread(osp.join(img_path, imgs[0]))
        new_annotations = {}
        id = 0
        for im in tqdm(imgs):
            path = osp.join(img_path, im)
            fname, _ = osp.splitext(im)
            image = cv2.imread(path)
            # print(annotations[im])
            bboxs = [[box[1][0], box[1][1], box[1][2], box[1][3], box[0]] for box in annotations[im]]
            H,W,C = image.shape
            for i,bbox in enumerate(bboxs):
                center_point = [bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2]
                x1 = int(center_point[0] - crop_h//2)
                x2 = int(center_point[0] + crop_h//2)
                y1 = int(center_point[1] - crop_w//2)
                y2 = int(center_point[1] + crop_w//2)
                x1 = x1 if x1>0 else 0
                x2 = x2 if x2<W else W
                y1 = y1 if y1>0 else 0
                y2 = y2 if y2<H else H
                crop = A.Compose(
                    [A.Crop(x1,y1,x2,y2)],
                    bbox_params=A.BboxParams(format='coco')
                )
                # import pdb
                # pdb.set_trace()
                transformed = crop(image=image, bboxes=bboxs)
                new_image = transformed['image']
                new_bbox = transformed['bboxes']
                new_name = fname + '_' + str(i) + '.png'
                new_box = [[box[4], box[:4]] for box in new_bbox]
                new_path = osp.join(save_path, new_name)
                new_im_info = [id, crop_h, crop_w]
                all_info = [new_box, new_im_info]
                new_annotations[new_name] = all_info
                cv2.imwrite(new_path, new_image)
                id += 1
        self.convert2coco(new_annotations, json_path)
    def crop_image_val(self,img_path,annotations,crop_h,crop_w,aug_num,json_name,im_fold):
        '''
        img_path:需要切割的图片的路径
        annotations:图片的标注，格式为
        {
            img_name:[category,[bbox]]
        }
        crop_h:切割图片的高度
        crop_w:切割图片的宽度
        aug_num:需要增广的数量
        '''
        # im2ann_dict, im_name2im_id_dict = self.load_json(ann_path)
        save_path = osp.dirname(img_path)
        json_path = osp.join(save_path,json_name)
        save_path = osp.join(save_path, im_fold)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        imgs = os.listdir(img_path)
        crop = A.Compose(
            [A.RandomCrop(crop_h,crop_w)],
            bbox_params=A.BboxParams(format='coco')
        )
        # image = cv2.imread(osp.join(img_path, imgs[0]))
        new_annotations = {}
        id = 0
        for im in tqdm(imgs):
            path = osp.join(img_path,im)
            fname,_ = osp.splitext(im)
            image = cv2.imread(path)
            # print(annotations[im])
            bbox = [[box[1][0],box[1][1],box[1][2],box[1][3],box[0]] for box in annotations[im]]
            for i in range(aug_num):
                transformed = crop(image=image,bboxes = bbox)
                new_image = transformed['image']
                new_bbox = transformed['bboxes']
                new_name = fname+'_'+str(i)+'.png'
                new_box = [[box[4],box[:4]] for box in new_bbox]
                new_path = osp.join(save_path,new_name)
                new_im_info = [id,crop_h,crop_w]
                all_info = [new_box,new_im_info]
                new_annotations[new_name] = all_info
                cv2.imwrite(new_path,new_image)
                id+=1
        self.convert2coco(new_annotations,json_path)

    def crop_image_train_new(self,img_path,annotations,crop_h,crop_w,json_name,im_fold):
        # import pdb
        # pdb.set_trace()
        save_path = osp.dirname(img_path)
        json_path = osp.join(save_path, json_name)
        save_path = osp.join(save_path, im_fold)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        imgs = os.listdir(img_path)
        # crop = A.Compose(
        #     [A.RandomCrop(crop_h, crop_w)],
        #     bbox_params=A.BboxParams(format='coco')
        # )
        # image = cv2.imread(osp.join(img_path, imgs[0]))
        new_annotations = {}
        id = 0
        for im in tqdm(imgs):
            path = osp.join(img_path, im)
            fname, _ = osp.splitext(im)
            image = cv2.imread(path)
            # print(annotations[im])
            bboxs = [[box[1][0], box[1][1], box[1][2], box[1][3], box[0]] for box in annotations[im]]
            H,W,C = image.shape
            # for i,bbox in enumerate(bboxs):
            for x in range(0,W-crop_w,200):
                for y in range(0,H-crop_h,200):
                    x1 = int(x)
                    x2 = int(x+crop_w)
                    y1 = int(y)
                    y2 = int(y+crop_h)
                    # x1 = x1 if x1>0 else 0
                    # x2 = x2 if x2<W else W
                    # y1 = y1 if y1>0 else 0
                    # y2 = y2 if y2<H else H
                    crop = A.Compose(
                        [A.Crop(x1,y1,x2,y2)],
                        bbox_params=A.BboxParams(format='coco')
                    )
                    # import pdb
                    # pdb.set_trace()
                    transformed = crop(image=image, bboxes=bboxs)
                    new_image = transformed['image']
                    new_bbox = transformed['bboxes']
                    # import pdb
                    # pdb.set_trace()
                    # pixle_value = new_image.sum()
                    # print(pixle_value)
                    if new_bbox:
                        new_name = fname + '_' + str(id) + '.png'
                        new_box = [[box[4], box[:4]] for box in new_bbox]
                        new_path = osp.join(save_path, new_name)
                        new_im_info = [id, crop_h, crop_w]
                        all_info = [new_box, new_im_info]
                        new_annotations[new_name] = all_info
                        cv2.imwrite(new_path, new_image)
                        id += 1
                    else:
                        continue
        self.convert2coco(new_annotations, json_path)


    def crop_image_test(self,img_path,save_im):
        '''
        对测试集进行切割，切割为128*128的小图,注意，文件夹中图片尺寸应该相同
        :param img_path:
        :return: None
        '''
        save_path = osp.dirname(img_path)
        save_path = osp.join(save_path,save_im)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        imgs = os.listdir(img_path)
        image = cv2.imread(osp.join(img_path,imgs[0]))
        crop_h = image.shape[0]//500
        crop_w = image.shape[1]//500
        for img in tqdm(imgs):
            fname,ext = osp.splitext(img)
            path = osp.join(img_path,img)
            image = cv2.imread(path)
            for i in range(crop_h):
                x0 = i*500
                x1 = (i+1)*500
                for j in range(crop_w):
                    y0 = j*500
                    y1 = (j+1)*500
                    croped_image = image[x0:x1,y0:y1]
                    new_name = fname+'_'+str(i)+'_'+str(j)+ext
                    new_path = osp.join(save_path,new_name)
                    cv2.imwrite(new_path,croped_image)

    def crop_image_test_new(self,img_path,save_im,cam):
        '''
        对测试集进行切割，切割为128*128的小图,注意，文件夹中图片尺寸应该相同
        :param img_path:
        :return: None
        '''
        thresh = 0
        if cam=="CAM1":
            thresh = 80
        elif cam == "CAM2":
            thresh = 40
        elif cam=="CAM3":
            thresh = 40
        save_path = osp.dirname(img_path)
        save_path = osp.join(save_path,save_im)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        imgs = os.listdir(img_path)
        image = cv2.imread(osp.join(img_path,imgs[0]))
        H,W,C = image.shape
        for img in tqdm(imgs):
            fname,ext = osp.splitext(img)
            path = osp.join(img_path,img)
            image = cv2.imread(path)
            for x in range(0,W-640,400):
                # print(x)
                x0 = x
                x1 = x0 + 640
                # print(x0,x1)
                for y in range(0,H-640,400):
                    y0 = y
                    y1 = y0+640
                    croped_image = image[y0:y1,x0:x1]
                    if croped_image.mean()>thresh:
                    # import pdb
                    # pdb.set_trace()
                        new_name = fname+'_'+str(x)+'_'+str(y)+ext
                        new_path = osp.join(save_path,new_name)
                        cv2.imwrite(new_path,croped_image)

    def devide_label(self,annotations_bbox,annotations_img):
        '''

        :param annotations_bbox: 图片名到bbox的字典
        :param annotations_img: 图片名到图片信息的字典包括图片id以及长宽
        :return: 三种相机图片的标注，格式如下
        {
         file_name:[bbox,[id,width,height]]
        }
        '''

        cam1_ann = {}
        cam2_ann = {}
        cam3_ann = {}
        for ann in annotations_bbox.keys():
            if 'CAM1' in ann:
                cam1_ann[ann] = [annotations_bbox[ann],annotations_img[ann]]
            elif 'CAM2' in ann:
                cam2_ann[ann] = [annotations_bbox[ann],annotations_img[ann]]
            elif 'CAM3' in ann:
                cam3_ann[ann] = [annotations_bbox[ann],annotations_img[ann]]
        return cam1_ann,cam2_ann,cam3_ann

    def _xywh2xyxy(self,bbox,require_int = True):
        bbox = [int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])] if require_int else \
            [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        return bbox

    def convert2coco(self,imname_to_all_info,save_path):
        '''
        将得到的标注转换成coco格式,annotations格式：

        :param annotations:
        :return:
        '''
        categories = [
            {
                'id': 1,
                'name': 'edge'
            },
            {
                'id': 2,
                'name': 'angle'
            },
            {
                'id': 3,
                'name': 'white_point'
            },
            {
                'id': 4,
                'name': 'shallow_color'
            },
            {
                'id': 5,
                'name': 'deep_color'
            },
            {
                'id': 6,
                'name': 'aperture'
            }

        ]
        coco = {}
        coco['categories'] = categories
        images = []
        annotaions = []
        id = 0
        for im_name in tqdm(imname_to_all_info.keys()):
            img ={}
            all_info = imname_to_all_info[im_name]
            anno_info = all_info[0]
            image_info = all_info[1]
            img['id'] = image_info[0]
            img['width'] = image_info[1]
            img['height'] = image_info[2]
            img['file_name'] = im_name
            images.append(img)
            for anno in anno_info:
                ann={}
                ann['image_id'] = image_info[0]
                ann['category_id'] = anno[0]
                ann['bbox'] = anno[1]
                ann['iscrowd'] = 0
                ann['area'] = anno[1][2]*anno[1][3]
                ann['id'] = id
                id +=1
                annotaions.append(ann)
        coco['images'] = images
        coco['annotations'] = annotaions
        self.save_json(coco,save_path)
    def save_json(self,obj,file_name):
        with open(file_name,'w') as f:
            json.dump(obj,f,indent=6)

    def extrace_flaw(self,img_root,annotations):
        '''
        将原本大图的所有标注缺陷都提取出来，并且存在文件夹中，方便之后增广
        :param img_root: 大图的位置
        :param annotations: 大图的标注
        :return:
        '''
        flaw_fold = osp.join(osp.dirname(img_root),'flaw_img')
        if not osp.exists(flaw_fold):
            os.makedirs(flaw_fold)
        for img_name in tqdm(annotations.keys()):
            fname,ext = osp.splitext(img_name)
            image = cv2.imread(osp.join(img_root,img_name))
            bboxs = annotations[img_name][0]
            for i,bbox in enumerate(bboxs):
                category = bbox[0]
                box = bbox[1]
                # box = [int(box[0]),int(box[1]),int(box[0]+box[2]),int(box[1]+box[3])]
                box = self._xywh2xyxy(box)
                class_path = osp.join(flaw_fold,str(category))
                if not osp.exists(class_path):
                    os.makedirs(class_path)
                flaw = image[box[1]:box[3],box[0]:box[2]]
                flaw_name = fname+'_'+str(category)+'_'+str(i)+ext
                flaw_path = osp.join(class_path,flaw_name)
                cv2.imwrite(flaw_path,flaw)
                # cv2.imshow('src',flaw)
                # cv2.waitKey()

    def aug_data(self,origin_image,save_path,annotations_bbox,annotations_img):
        '''
        对缺陷做增广,并且更新标注文件
        {
       "0": "背景",
       "1": "边异常",
       "2": "角异常",
       "3": "白色点瑕疵",
       "4": "浅色块瑕疵",
       "5": "深色点块瑕疵",
       "6": "光圈瑕疵"
      }
        origin_image:原始图片的位置
        flaw_image:提取出来的缺陷的位置
        annotations：原图的数据标注
        :return:
        '''
        import copy
        annotations = {im_name:[annotations_bbox[im_name],annotations_img[im_name]] for im_name in annotations_bbox.keys()}
        transform_3456 = A.Compose([ #对第3,4,5,6类缺陷做增广
            A.RandomScale((1.0,1.2)),
            A.Flip(),
            # A.GridDistortion(distort_limit=0.3,num_steps = 5),
            A.RandomRotate90()
        ])
        transform_angle = A.Compose([ #对第1，2类缺陷做增广
            A.RandomScale(),
            # A.elastic_transform(),
        ])

        image_list = os.listdir(origin_image)
        # flaw_1 = osp.join(flaw_image,'1')
        # flaw_2 = osp.join(flaw_image,'2')
        # flaw_3 = osp.join(flaw_image,'3')
        # flaw_4 = osp.join(flaw_image,'4')
        # flaw_5 = osp.join(flaw_image,'5')
        # flaw_6 = osp.join(flaw_image,'6')
        json_save_path = osp.join(osp.dirname(origin_image),'auged_normal.json')
        # import pdb
        # pdb.set_trace()
        annotations_new = copy.deepcopy(annotations)
        for img in tqdm(image_list):
            img_path = osp.join(origin_image,img)
            # ann_bbox = annotations_bbox[img]
            anno = annotations[img]
            ann_bbox = anno[0]
            # ann_img = annotations_img[img]
            image = cv2.imread(img_path)
            new_path = osp.join(save_path,img)
            for box in ann_bbox:
                # import pdb
                # pdb.set_trace()
                bbox = box[1]
                bbox = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
                width = int(bbox[2])
                height = int(bbox[3])
                area = width*height
                category = box[0]
                if 0<area<1024:
                    for i in range(10):
                        # import pdb
                        # pdb.set_trace()
                        flaw = image[bbox[1]:bbox[1]+height,bbox[0]:bbox[0]+width]
                        if height<=0 or width <=0:
                            import pdb
                            pdb.set_trace()
                        flaw = transform_3456(image=flaw)['image']
                        f_h,f_w,_ = flaw.shape
                        # mask = 255 * np.ones((f_h, f_w, 3), dtype=np.uint8)
                        w, h = self.random_coor(0, 500 - f_w - 1, 0, 500 - f_h - 1,image)
                        # image = cv2.seamlessClone(flaw, image, mask, (w + f_w // 2,h + f_h // 2, ),
                        #                               cv2.MIXED_CLONE)  # 泊松融合

                        image[h:h+f_h,w:w+f_w] = flaw
                        new_bbox = [w,h,f_w,f_h]
                        annotations_new[img][0].append([category,new_bbox])
            cv2.imwrite(new_path,image)

            # new_image = self.aug_3456(img_path,flaw_3,flaw_4,flaw_5,flaw_6,annotations,transform_3456)
            # new_path = osp.join(save_path,img)
            # cv2.imshow('src',new_image)
            # cv2.waitKey()
            # cv2.imwrite(new_path,new_image)
            # self.aug_angle(img_path,flaw_2,annotations,transform_angle)
            # self.aug_edge(img_path,flaw_1,annotations,transform_angle)
        self.convert2coco(annotations_new, json_save_path)
    def aug_3456(self,origin_image,flaw3,flaw4,flaw5,flaw6,annotations,transform):
        '''
        对第3，4，5，6种缺陷做增广
        :return:
        '''
        # global image
        image = cv2.imread(origin_image)
        img_name = osp.basename(origin_image)
        # min_x,max_x,min_y,max_y = self.find_angle(image)
        H = image.shape[0]
        W = image.shape[1]
        # image[0,:] = 255
        # image[H-1,:] = 255
        # image[:,0] = 255
        # image[:,W-1] = 255
        min_x = H//15 #宽度
        max_x = H//15*14
        min_y = W//6 #高度
        max_y = W//6*5
        flaw3_list = os.listdir(flaw3)
        flaw4_list = os.listdir(flaw4)
        flaw5_list = os.listdir(flaw5)
        flaw6_list = os.listdir(flaw6)
        # for i in range(10):
        image = self.aug_single(flaw3,flaw3_list,transform,min_x,max_x,min_y,max_y,img_name,annotations,3,image)
        # image = self.aug_single(flaw4,flaw4_list,transform,min_x,max_x,min_y,max_y,img_name,annotations,4,image)
        # image = self.aug_single(flaw5,flaw5_list,transform,min_x,max_x,min_y,max_y,img_name,annotations,5,image)
        # image = self.aug_single(flaw6,flaw6_list,transform,min_x,max_x,min_y,max_y,img_name,annotations,6,image)
        return image


    def aug_single(self,flaw,flaw_list,transform,min_x,max_x,min_y,max_y,im_name,annotations,category,image):
        '''

        :param flaw:  缺陷图片目录
        :param flaw_list: 缺陷图片名列表
        :param transform: 数据增广方式
        :param H: 原图高度
        :param W: 原图宽度
        :param image: 原图
        :param im_name: 原图名
        :param annotations: 图片标注
        :param category: 增广的缺陷类别
        :return:
        '''
        # global image
        import copy
        auged_img = copy.copy(image)
        # for i in range(10):
        flaw_name = random.choice(flaw_list)
        flaw_path = osp.join(flaw, flaw_name)
        flaw_array = cv2.imread(flaw_path)
        flaw_array = transform(image = flaw_array)['image']
        h,w = self.random_coor(min_x,max_x,min_y,max_y)
        height,width = flaw_array.shape[0],flaw_array.shape[1]
        if height>256 or width>256:
            flaw_array = cv2.resize(flaw_array,(256,256))
            height = width = 256
        # mask = 255 * np.ones(flaw_array.shape,dtype=np.uint8)
        mask = 255*np.ones((height,width,3),dtype = np.uint8)

        # mask[0,:] = 0
        # mask[height,:] = 0
        # mask[:,0] = 0
        # mask[:,width] = 0

        # import pdb
        # pdb.set_trace()
        auged_img = cv2.seamlessClone(flaw_array,auged_img,mask,(h+height//2,w+width//2),cv2.MIXED_CLONE) #泊松融合
        # image[h:h+height,w:w+width] = flaw_array
        annotations[im_name][0].append([category,[w,h,width,height]])
        return image

    def random_coor(self,min_w,max_w,min_h,max_h,image):
        i = 0
        h = random.randint(min_h,max_h)
        w = random.randint(min_w,max_w)
        piece = image[h:h+32,w:w+32]
        # print(piece.sum())
        while piece.sum()<100000:
            h = random.randint(min_h, max_h)
            w = random.randint(min_w, max_w)
            piece = image[h:h + 32, w:w + 32]
        #     i+=1
        #     if 10<=i:
        #         break
        return w,h

    def find_angle(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, 3, 3, 0.01)
        dst = dst >0.01*dst.max()
        corner = np.where(dst==True)
        print(len(corner[0]))
        corner[0].sort()
        corner[1].sort()
        min_x = corner[0][int(0.1*len(corner[0]))]
        max_x = corner[0][int(0.9*len(corner[0]))]
        min_y = corner[1][int(0.1*len(corner[0]))]
        max_y = corner[1][int(0.9*len(corner[0]))]
        # corner = [(x,y) for x,y in zip(corner[0],corner[1])]
        # center_coor = random.choice(corner)
        return (min_x,max_x,min_y,max_y)
        # image[dst > 0.01 * dst.max()] = [0, 0, 255]



    def aug_angle(self,origin_image,flaw,annotations,transform):
        '''
        对角缺陷做增广
        :return:
        '''
        image = cv2.imread(origin_image)
        H,W = image.shape[0],image.shape[1]
        img_name = osp.basename(origin_image)
        flaw_list = os.listdir(flaw)
        for i in range(50):
            flaw_name = random.choice(flaw_list)
            flaw_path = osp.join(flaw,flaw_name)
            flaw_array = cv2.imread(flaw_path)
            flaw_array = transform(flaw_array)
            height,width = flaw_array.shape[0],flaw_array.shape[1]
            # corner_point = self.find_angle(image)
            h,w = self.random_coor(H//6,H//6*5,W//15,W//15*14)
            # h = h - height
            # w = w - width
            image[h:h+height,w:w+width] = flaw_array
            annotations[img_name] .append([2,[h,w,height,width]])

    def aug_edge(self,origin_image,flaw,annotations,transform):
        '''
        对边缘缺陷做增广
        :return:
        '''
        pass

def generate_test_anno(test_img_path,save_path):
    categories = [
        {
            'id': 1,
            'name': 'edge'
        },
        {
            'id': 2,
            'name': 'angle'
        },
        {
            'id': 3,
            'name': 'white_point'
        },
        {
            'id': 4,
            'name': 'shallow_color'
        },
        {
            'id': 5,
            'name': 'deep_color'
        },
        {
            'id': 6,
            'name': 'aperture'
        }

    ]
    coco = {}
    coco['categories'] = categories
    images = []
    annotaions = []
    image_list = os.listdir(test_img_path)
    for i,im in tqdm(enumerate(image_list)):
        img = {}
        img['id'] = i
        img['width'] = 500
        img['height'] = 500
        img['file_name'] = im
        images.append(img)
    coco['images'] = images
    coco['annotations'] = annotaions
    with open(save_path,'w') as f:
        json.dump(coco,f,indent=6)


def statistic_dark_area(image_path):
    imgs  = os.listdir(image_path)
    result = []
    for img in tqdm(imgs):
        path = osp.join(image_path,img)
        image = cv2.imread(path)
        value = image[:128,:128].mean()
        result.append(value)
    return np.mean(result)


if __name__=="__main__":
    args = parse_args()
    image = cv2.imread('/home/dell/桌面/test_code_imgs/imgs/197_1_t20201119084916148_CAM1.jpg')
    angle_detection(image)
    # json_path = './tile_round1_train_20201231/coco_anno.json'
    # train_path = './tile_round1_train_20201231/train_imgs'
    # test_path = '/home/xiongpan/dataset/flaw-detection/tile_round1_testA_20201231/testA_imgs'
    # process_data = ProcessData(json_path, test_path, train_path)
    # test_crop_path_1 = 'crop'
    # test_path_1 = '../tile_round1_testA_20201231/cam1/origin'
    # test_crop_path_2 = 'crop'
    # test_path_2 = '../tile_round1_testA_20201231/cam2/origin'
    # test_crop_path_3 = 'crop'
    # test_path_3 = '../tile_round1_testA_20201231/cam3/origin'

    # process_data.crop_image_test(test_path_1,test_crop_path_1)
    # process_data.crop_image_test(test_path_2,test_crop_path_2)
    # process_data.crop_image_test(test_path_3,test_crop_path_3)
    # #原图json位置
    # json_path_cam1 = '/home/xiongpan/dataset/flaw-detection/tile_round1_train_20201231/tile_round1_train_20201231/cam1/cam1.json'
    # json_path_cam2 = '/home/xiongpan/dataset/flaw-detection/tile_round1_train_20201231/tile_round1_train_20201231/cam2/cam2.json'
    # json_path_cam3 = '/home/xiongpan/dataset/flaw-detection/tile_round1_train_20201231/tile_round1_train_20201231/cam3/cam3.json'
    # # # train_path_cam1 = './tile_round1_train_20201231/cam1/train_img'
    # # # train_path_cam2 = './tile_round1_train_20201231/cam2/train_img'
    # # # train_path_cam3 = './tile_round1_train_20201231/cam3/train_img'
    # im2ann_dict1, im_name2im_id_dict1 = process_data.load_json(json_path_cam1)
    # im2ann_dict2, im_name2im_id_dict2 = process_data.load_json(json_path_cam2)
    # im2ann_dict3, im_name2im_id_dict3 = process_data.load_json(json_path_cam3)
    # # # process_data.crop_image_train(train_path_cam1,im2ann_dict1,500,500,5,'val.json','val')
    # # # process_data.crop_image_train(train_path_cam2,im2ann_dict2,500,500,5,'val.json','val')
    # # # process_data.crop_image_train(train_path_cam3,im2ann_dict3,500,500,5,'val.json','val')
    # # # process_data.devide_label(im2ann_dict,im_name2im_id_dict)
    # # # # process_data.devide_image(train_path)
    # # # # process_data.devide_image(test_path)
    # # # cam1_ann,cam2_ann,cam3_ann = process_data.devide_label(im2ann_dict,im_name2im_id_dict)
    # #原图图片位置
    # cam1_img_path = '/home/xiongpan/dataset/flaw-detection/tile_round1_train_20201231/tile_round1_train_20201231/cam1/train_img'
    # cam2_img_path = '/home/xiongpan/dataset/flaw-detection/tile_round1_train_20201231/tile_round1_train_20201231/cam2/train_img'
    # cam3_img_path = '/home/xiongpan/dataset/flaw-detection/tile_round1_train_20201231/tile_round1_train_20201231/cam3/train_img'
    # # cam1_flaw = osp.join(osp.dirname(cam1_img_path),'flaw_img')
    # cam2_flaw = osp.join(osp.dirname(cam2_img_path),'flaw_img')
    # cam3_flaw = osp.join(osp.dirname(cam3_img_path),'flaw_img')
    # cam1_flaw_save = osp.join(osp.dirname(cam1_img_path),'auged_data_normal')
    # cam2_flaw_save = osp.join(osp.dirname(cam2_img_path),'auged_data_normal')
    # cam3_flaw_save = osp.join(osp.dirname(cam3_img_path),'auged_data_normal')
    # if not osp.exists(cam1_flaw_save):
    #     os.makedirs(cam1_flaw_save)
    # if not osp.exists(cam2_flaw_save):
    #     os.makedirs(cam2_flaw_save)
    # if not osp.exists(cam3_flaw_save):
    #     os.makedirs(cam3_flaw_save)
    # process_data.aug_data(cam1_img_path,cam1_flaw_save,im2ann_dict1,im_name2im_id_dict1)
    # process_data.aug_data(cam2_img_path,cam2_flaw_save,im2ann_dict2,im_name2im_id_dict2)
    # process_data.aug_data(cam3_img_path,cam3_flaw_save,im2ann_dict3,im_name2im_id_dict3)
    # process_data.crop_image_train_new(cam1_img_path,im2ann_dict1,640,640,'croped_new.json','croped_new')
    # process_data.crop_image_train_new(cam2_img_path,im2ann_dict2,640,640,'croped_new.json','croped_new')
    # process_data.crop_image_train_new(cam3_img_path,im2ann_dict3,640,640,'croped_new.json','croped_new')
    # test_path_cam1 = '/home/xiongpan/dataset/flaw-detection/tile_round1_testA_20201231/cam1/origin'
    # test_path_cam2 = '/home/xiongpan/dataset/flaw-detection/tile_round1_testA_20201231/cam2/origin'
    # test_path_cam3 = '/home/xiongpan/dataset/flaw-detection/tile_round1_testA_20201231/cam3/origin'
    # process_data.crop_image_test_new(test_path_cam1,'croped_test','CAM1')
    # process_data.crop_image_test_new(test_path_cam2,'croped_test','CAM2')
    # process_data.crop_image_test_new(test_path_cam3,'croped_test','CAM3')
    # thre1 = statistic_dark_area(cam1_img_path)
    # thre2 = statistic_dark_area(cam2_img_path)
    # thre3 = statistic_dark_area(cam3_img_path)
    # print(thre1,thre2,thre3)
    # process_data.convert2coco(cam1_ann,cam1_json_path)
    # process_data.convert2coco(cam2_ann,cam2_json_path)
    # process_data.convert2coco(cam3_ann,cam3_json_path)
    # process_data.extrace_flaw(cam1_img_path,cam1_ann)
    # process_data.extrace_flaw(cam2_img_path,cam2_ann)
    # process_data.extrace_flaw(cam3_img_path,cam3_ann)
    # test_img1 = '/tmp/tile_round1_train_20201231/cam1/test'
    # test_img2 = '/tmp/tile_round1_train_20201231/cam2/test'
    # test_img3 = '/tmp/tile_round1_train_20201231/cam3/test'
    # test_json1 = '/tmp/tile_round1_train_20201231/cam1/test.json'
    # test_json2 = '/tmp/tile_round1_train_20201231/cam2/test.json'
    # test_json3 = '/tmp/tile_round1_train_20201231/cam3/test.json'
    # generate_test_anno(test_img1,test_json1)
    # generate_test_anno(test_img2,test_json2)
    # generate_test_anno(test_img3,test_json3)
