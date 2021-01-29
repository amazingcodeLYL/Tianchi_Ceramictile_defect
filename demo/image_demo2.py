from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import json
import numpy

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.000000001, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    # import pdb;
    # pdb.set_trace()
    #

    test_img_dir="/home/dell/桌面/tile_round1_testA_20201231/tile_round1_testA_20201231/cam2/crop/"
    imgs_dir=os.listdir(test_img_dir)
    json_result=[]
    for img in imgs_dir:
        img_name=img
        img=test_img_dir+img
        result = inference_detector(model, img)
        # show the results
        show_result_pyplot(model,result,img,img_name,json_result=json_result,score_thr=args.score_thr)


    with open("/home/dell/桌面/guangdong/result_cam2.json", "w") as f:
        json.dump(json_result, f,cls=MyEncoder,indent=6)
    print("加载入文件完成...")



######################转换写入json类型
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == '__main__':
    main()
