import os

dataset_dir = "./mouse"
# list_1 = os.listdir(dataset_dir)
# print(list_1)
for category_dir in os.listdir(dataset_dir):   # 第一层是train和val目录
    category_path = os.path.join(dataset_dir,category_dir)
    print(category_path)
    for img_name in os.listdir(category_path):   # 里面遍历到所有图片名称
        pure_img_name = img_name.split('.')[0]   # 纯文件名
        img_path = os.path.join(category_path, img_name)  # 图片路径
        print(img_path)
        if category_dir.split('_')[0] == 'train':  # 训练集如果最后为0就删除
            if pure_img_name[-1] == '0':
                os.remove(img_path)
                print('remove {}'.format(img_path))

        elif category_dir.split('_')[0] == 'val':   # 测试集如果最后不为0就删除
            if pure_img_name[-1] != '0':
                os.remove(img_path)
                print('remove {}'.format(img_path))



