import os
from PIL import Image
import numpy as np

img_dir = "./web_SA/images"
img_dir = './SP_cyclegan_test/SP_cyclegan_org/test_120/images'

# store_dir = "./web_res9/images_output"
for image in os.listdir(img_dir):   # 第一层是train和val目录
    print(image)
    image_path = os.path.join(img_dir, image)
    # image_store_path = os.path.join(store_dir, image)
    img = Image.open(image_path)
    np_img = np.array(img)
    # np_img_mean = np.expand_dims(np_img.mean(axis=2), axis=2)
    np_img_mean = np_img.mean(axis=2)
    img_mean = Image.fromarray(np.uint8(np_img_mean))
    # img_mean.save(image_store_path)
    img_mean.save(image_path)
    pass
