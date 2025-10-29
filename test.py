# This is a script for performing etm-rl operations on images.
# If you want to obtain images with different compression rates, please modify the r value.

import timm
import tome
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

import numpy as np
olderr = np.seterr(all='ignore')
model_name = "vit_large_patch16_384"
model = timm.create_model(model_name, pretrained=True)
tome.patch.timm(model, trace_source=True)
input_size = model.default_cfg["input_size"][1]

# Make sure the transform is correct for your model!
transform_list = [
    transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(input_size)
]

# The visualization and model need different transforms
transform_vis  = transforms.Compose(transform_list)
transform_norm = transforms.Compose(transform_list + [
    transforms.ToTensor(),
    transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
])

# Enter the address of the image you want to process
img = Image.open(r"E:/ETM-RL-main/examples/images/4.JPEG").convert('RGB')
img_vis = transform_vis(img)
img_norm = transform_norm(img)


#model.r = [25] * 14   # 377 tokens at the end
#model.r = [25] * 16  # 177 tokens at the end
#model.r = [25] * 22  # 27 tokens at the end
model.r = 25 # 8 tokens at the end
_ = model(img_norm[None, ...])
source = model._tome_info["source"]

# image = _ .cpu().clone()  # we clone the tensor to not do changes on it
# image = image.squeeze(0)  # remove the fake batch dimension
# image = matplotlib.unloader(image)
# matplotlib.plt.imshow(image)
print(f"{source.shape[1]} tokens at the end")
vis_image = tome.make_visualization(img_vis, source, patch_size=16, class_token=True)
vis_image.show()


