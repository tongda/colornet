import matplotlib.pyplot as plt
import glob
import os

def filter_out_small_images(path):
    filenames = glob.glob(path)
    for fn in filenames:
        img = plt.imread(fn)
        if img.shape[0] < 224 or img.shape[1] < 224:
            print("found small image {}, removing it.".format(fn))
            os.remove(fn)

if __name__ == "__main__":
    filter_out_small_images("./images/imagenet/*.JPEG")
