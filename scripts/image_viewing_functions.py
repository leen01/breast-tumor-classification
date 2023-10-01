import tensorflow as tf

def imshow(image, title=None):
  """ Show image from tensor """
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
    
def clip_0_1(image):
  """ Ensure values for image are between 0 and 1"""
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def create_image_set(ds, num_images = 5): 
    """ Create preview of images with labels 
    Takes the num_images from the dataset for viewing in plots
    
    Args: 
    ds (tensorflow dataset): tensorflow dataset with images
    """
    
    images = []
    for i, t in enumerate(ds.take(num_images)): 
        images.append(
            {'label': tf.argmax(t[1].numpy()).numpy(), 
            'im' : t[0].numpy()}
        )

    return images


def create_image_set_batched(ds, batch_size = 32, num_images = 1): 
    """ Create preview of images with labels 
    Takes the num_images from the dataset for viewing in plots. meant to work with batched data
    
    Args: 
    ds (tensorflow dataset): tensorflow dataset with images
    """
    
    images = []
    for i, t in enumerate(ds.take(num_images)):
        test_labels = tf.argmax(t[1], axis = 1).numpy()
        for b in range(batch_size):
            images.append(
                {'label': test_labels[b], 
                'im' : t[i][b].numpy()}
            )
    return images


####### 
# * ploting functions * # 
###### 

import matplotlib.pyplot as plt


def plot_sobel_images(images): 
    """ List of image dictionaries with label: value, im: array
    Plots the sobel filter applied to an image side by side for the vertical and horizontal edges. 
    
    """
    
    nrows = int(len(images))
    ncols = 2

    fig, axs = plt.subplots(nrows, ncols, figsize=(5, 10))
    
    def squeeze_3d(image, title=None):
        """ Show image from tensor"""
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
        return image

    for i, im in enumerate(images):
        # set label for the graph
        label = str(im['label'])
        
        # extract image array
        im = im['im']
        
        # two, one for the horizontal and vertical sobel edges
        for j in range(2): 
            # add image to axes            
            axs[i,j].imshow(squeeze_3d(im[..., j]/4+0.5))
            
            # drop axis labels
            axs[i,j].axis('off')
            
            # add axes title
            if j ==  0: 
                title = 'Horizontal'
            else: 
                title = 'Vertical'
                
            axs[i,j].set(title="-".join([label,title]))
        
    fig.suptitle("Sample of Sobel-edges", fontsize=16)
    
    plt.tight_layout()

    plt.show()


def plot_images(images, nrows, ncols, gray_scale = False): 
    """ List of image dictionaries with label: value, im: array"""

    if gray_scale == True: 
        plt.set_cmap("gray")
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(8, 3))    
    fig.text(.5, .0001, "Sample of images from dataset", ha='center')

    if nrows == 1 or ncols == 1:
        plots = max(nrows, ncols)
        for i in range(plots):
            label = str(images[i]['label'])
            
            im = images[i]['im']

            # add image to axes
            axs[i].imshow(im)
            # drop axis labels
            axs[i].axis('off')
            # add axes title
            axs[i].set(title=label)
            
    else: 
        # create iterator to continuously index the images
        im_index = 0
        for r in range(nrows):
            for c in range(ncols):
                # label for subplot title
                label = str(images[im_index]['label'])
                
                # image array
                im = images[im_index]['im']

                # add image to axes
                axs[r,c].imshow(im)
                
                # drop axis labels
                axs[r,c].axis('off')
                
                # add axes title
                axs[r,c].set(title=label)
                
                # next index
                im_index += 1
                
    plt.tight_layout()
    plt.show();
    
def plot_images_with_histogram(images, nrows): 
    """ List of image dictionaries with label: value, im: array
    plots image with histogram side by side
    """
    
    fig, axs = plt.subplots(nrows, ncols = 2,  figsize=(20, 20), layout="constrained")

    # Flatten axes to make it easier to access
    # axs = axs.flatten()

    for i, im in enumerate(images):
        #read in file
        # pic = plt.imread(file)
        label = str(im['label'])
        
        im = im['im']
        # add image to axes
        axs[i, 0].imshow(im)
        # drop axis labels
        axs[i, 0].axis('off')
        # add axes title
        axs[i, 0].set(title=label)
        
        # add histogram 
        axs[i, 1].hist((im*255).ravel(), bins = 100, fc='k', ec='k')
        
    fig.tight_layout()
    
    plt.show()
