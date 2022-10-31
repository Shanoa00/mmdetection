import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '/home/mauricio/Documents/Pytorch/mmdetection/code/mmdetection/work_dirs/swin_nancho_pre_kuzh/exported.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
#onnx_model = onnx.load(onnx_model_path)
#torch_model_2 = convert(onnx_model)
model = torch.load(torch_model_1)
print(model)

# python3 tools/export.py /home/mauricio/Documents/Pytorch/mmdetection/code/mmdetection/work_dirs/swin_nancho_pre_kuzh/swin_nancho.py /home/mauricio/Documents/Pytorch/mmdetection/code/mmdetection/work_dirs/swin_nancho_pre_kuzh/best_mAP_epoch_15.pth /home/mauricio/Documents/Pytorch/mmdetection/code/mmdetection/work_dirs/swin_nancho_pre_kuzh/exp.onnx