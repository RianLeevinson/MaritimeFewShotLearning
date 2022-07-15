from email.mime import image
import os
from PIL import Image

#object_classes = ['buoys', 'harbour', 'human', 'large_commercial_vessel', 'leisure_craft', 'sailboats', 'small_medium_fishing_boat']
#num_classes = len(object_classes)

data_dir = r'C:\DTU\master_thesis\MaritimeFewShotLearning\data\processed\new_data_may\val_set_all'

object_classes = os.listdir(data_dir)
#class_to_idx = {classes[i]: i for i in range(len(classes))}

for object_class in object_classes:
    num_deleted_images = 0
    data_path = os.path.join(data_dir, object_class)
    for (root, dirs, files) in os.walk(data_path, topdown = True):
        for file in files:
            file_path = os.path.join(data_path, file)
            with Image.open(file_path) as img:
                img_size = img.size
            if img_size[0] < 64 and img_size[1] < 64:
                os.remove(file_path)
                num_deleted_images += 1
    print(f'The number of {object_class} images deleted is: ', num_deleted_images)     


# import os
# from PIL import Image

# #object_classes = ['buoys', 'harbour', 'human', 'large_commercial_vessel', 'leisure_craft', 'sailboats', 'small_medium_fishing_boat']
# #num_classes = len(object_classes)

# data_dir = r'C:\DTU\master_thesis\MaritimeFewShotLearning\data\processed\new_data_may\val_set'

# object_classes = os.listdir(data_dir)
# #class_to_idx = {classes[i]: i for i in range(len(classes))}

# for object_class in object_classes:
#     small_size = 0
#     img_list1 = []
#     large_size = 0
#     data_path = os.path.join(data_dir, object_class)
#     for (root, dirs, files) in os.walk(data_path, topdown = True):
#         for file in files:
#             file_path = os.path.join(data_path, file)
#             with Image.open(file_path) as img:
#                 img_size = img.size
#                 img_list1.append(img_size[0])
# img_list1.sort()
# print(img_list1[0])

#     #print(f'The number of {object_class} images deleted is: ', num_deleted_images)    
