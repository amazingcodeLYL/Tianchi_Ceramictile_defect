class Fabric2COCO:

    def __init__(self,
            is_mode = "train"
            ):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.is_mode = is_mode
        if not os.path.exists("/home/aistudio/PaddleDetection/dataset/coco/{}".format(self.is_mode)):
            os.makedirs("/home/aistudio/PaddleDetection/dataset/coco/{}".format(self.is_mode))

    def to_coco(self, anno_file,img_dir):
        self._init_categories()
        anno_result= pd.read_json(open(anno_file,"r"))
        
        if self.is_mode == "train":
            anno_result = anno_result.head(int(anno_result['name'].count()*0.9))#取数据集前百分之90
        elif self.is_mode == "val":
            anno_result = anno_result.tail(int(anno_result['name'].count()*0.1)) 
        name_list=anno_result["name"].unique()#返回唯一图片名字
        for img_name in name_list:
            img_anno = anno_result[anno_result["name"] == img_name]#取出此图片的所有标注
            bboxs = img_anno["bbox"].tolist()#返回list
            defect_names = img_anno["category"].tolist()
            assert img_anno["name"].unique()[0] == img_name

            img_path=os.path.join(img_dir,img_name)
            #img =cv2.imread(img_path)
            #h,w,c=img.shape
            #这种读取方法更快
            img = Image.open(img_path)
            w, h = img.size
            #h,w=6000,8192
            self.images.append(self._image(img_path,h, w))

            self._cp_img(img_path)#复制文件路径
            if self.img_id % 200 is 0:
                print("处理到第{}张图片".format(self.img_id))
            for bbox, label in zip(bboxs, defect_names):
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        #1，2，3，4，5，6个类别
        for v in range(1,7):
            print(v)
            category = {}
            category['id'] = v
            category['name'] = str(v)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    def _image(self, path,h,w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)#返回path最后的文件名
        return image

    def _annotation(self,label,bbox):
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        points=[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = []# np.asarray(points).flatten().tolist()
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation["ignore"] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join("/home/aistudio/PaddleDetection/dataset/coco/{}".format(self.is_mode), os.path.basename(img_path)))
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))#缩进设置为1，元素之间用逗号隔开 ， key和内容之间 用冒号隔开
            
  '''转换有瑕疵的样本为coco格式'''
#训练集,划分90%做为训练集
img_dir = "/home/aistudio/work/dataset/tile_round1_train_20201231/train_imgs"
anno_dir="/home/aistudio/work/dataset/tile_round1_train_20201231/train_annos.json"
fabric2coco = Fabric2COCO()
train_instance = fabric2coco.to_coco(anno_dir,img_dir)
if not os.path.exists("/home/aistudio/PaddleDetection/dataset/coco/annotations/"):
    os.makedirs("/home/aistudio/PaddleDetection/dataset/coco/annotations/")
fabric2coco.save_coco_json(train_instance, "/home/aistudio/PaddleDetection/dataset/coco/annotations/"+'instances_{}.json'.format("train"))
