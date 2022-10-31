
import mmcv
from mmdet.apis import inference_detector, show_result_pyplot
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

# Choose to use a config and initialize the detector
config= 'configs/kuzushiji.py'
# Setup a checkpoint file to load
checkpoint = 'work_dirs/hr32/latest.pth'
# Set the device to be used for evaluation
device='cuda:0'
# Load the config
config = mmcv.Config.fromfile(config)
# Initialize the detector
model = build_detector(config.model)
# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)
# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']
# We need to set the model's cfg for inference
model.cfg = config
# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

img = mmcv.imread('../../data/test_images/100241706_00020_1.jpg')
result = inference_detector(model, img)
show_result_pyplot(model, img, result)