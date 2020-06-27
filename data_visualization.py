# %%
# Import all the required module
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd


#%%
# Utility function
def load_image(path):
    img = Image.open(path)
    img.load()
    return np.asarray(img)


# Show image with the bounding boxes
def show_bounding_boxes(path, bounding_boxes, conf=1.0):
    """
    This function will print the bounding boxes inside the image
    :param path:
    :param bounding_boxes:
    :return:
    """
    fig, ax = plt.subplots(1)
    for bbox in bounding_boxes:
        rect = patches.Rectangle(xy=[bbox[0], bbox[1]], width=bbox[2], height=bbox[3],
                                 fill=False, edgecolor='red')
        ax.text(bbox[0], bbox[1], str(conf), va='bottom', ha='left', color='red')
        ax.add_patch(rect)
    ax.imshow(load_image(path))
    plt.show()


#%%
dataset = pd.read_csv("dataset/train.csv", converters={'bbox': eval})
single_image_id = dataset.iloc[0]['image_id']
single_image_path = "dataset/train/" + single_image_id + ".jpg"
# Get the bounding boxes of image with id `single_image_id`
dataset_single_image = dataset[dataset['image_id'] == single_image_id]
single_bounding_boxes = np.array(dataset_single_image["bbox"])
show_bounding_boxes(single_image_path, single_bounding_boxes)


