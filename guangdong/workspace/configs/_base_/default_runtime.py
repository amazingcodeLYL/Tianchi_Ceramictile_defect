checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from =  '/home/dell/桌面/guangdong/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume_from =None
workflow = [('train', 5)]
