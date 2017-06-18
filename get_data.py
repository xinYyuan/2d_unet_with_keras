import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import matplotlib.pyplot as plt
import h5py

working_path = "validation/"
file_list=glob(working_path+"images_*.npy")

def show_picture(data):
    plt.imshow(data,plt.cm.gray)
    plt.show()
if __name__ == '__main__':

    out_images = []
    out_mask = []
    for img_file in file_list:
        # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
        imgs_to_process = np.load(img_file).astype(np.float64)
        node_masks = np.load(img_file.replace("images", "masks"))
        #out_images = []
        #out_mask=[]
        #print "on image", img_file
        if imgs_to_process.shape[1]==512:
            for i in range(len(imgs_to_process)):
                img = imgs_to_process[i]
                node_mask = node_masks[i]
                # Standardize the pixel values

                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = (img-min)/ (max - min)

                out_images.append(img)
                out_mask.append(node_mask)

    num_images = len(out_images)
    final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
    final_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)

    for i in range(num_images):
            final_images[i, 0] = out_images[i]
            final_masks[i, 0] = out_mask[i]

    final_images = np.swapaxes(final_images, 1, 2)
    final_images = np.swapaxes(final_images, 3, 2)

    final_masks = np.swapaxes(final_masks,1,2)
    final_masks = np.swapaxes(final_masks, 3, 2)

    print final_images.shape
    print final_masks.shape

    show_picture(final_masks[0,:,:,0])
    show_picture(final_images[0,:,:,0])

    file = h5py.File('val_data_withoutseg.h5', 'w')
    file.create_dataset('data', data=final_images)

    file = h5py.File('val_label_withoutseg.h5', 'w')
    file.create_dataset('data', data=final_masks)


    #
    #    Here we're applying the masks and cropping and resizing the image
    #

