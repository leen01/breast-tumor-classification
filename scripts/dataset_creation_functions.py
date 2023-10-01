# necessary packages
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

AUTOTUNE = tf.data.AUTOTUNE

def create_tf_ds(file_paths, labels):
    """ Create tensorflow dataset from file paths and labels """
    # create dataset
    ds = tf.data.Dataset.from_tensor_slices(
        (file_paths, labels))  # mapping like a zip in python

    return ds

def augment_crop(image, label, height = 224, width = 224): 
    """ Crops image by defining a bounding box
    
    Using to make sure the images are the same size.     
    """
    # normalize image size to a square and get smaller subsection
    image = tf.image.random_crop(value=image, size=(height, width, 3))
    
    return image, label


### Read in images ### 
def read_image(image_file_path, label):
    """Function to read in images to create tf dataset"""
    
    image = tf.io.read_file(image_file_path)  # read in image
    
    # preserving RGB colors with channels = 3. Reads array
    image = tf.image.decode_image(image, channels=3, dtype=tf.float64)
    
    image, label = augment_crop(image,label)

    return image, label

def read_image2(image_file_path, label):
    """Function to read in images to create tf dataset. Adds an extra dimension for batch. Allows use on sobel filter. """
    
    image = tf.io.read_file(image_file_path)  # read in image
    
    # preserving RGB colors with channels = 3. Reads array
    image = tf.image.decode_image(image, channels=3, dtype=tf.float64)
    
    # add batch dimension
    image = image[tf.newaxis, :]

    return image, label

def read_image_binary(image_file_path, label): 
    """Function to read in images to create tf dataset. Adds an extra dimension for batch. Allows use on sobel filter. """
    
    image = tf.io.read_file(image_file_path)  # read in image
    
    # preserving RGB colors with channels = 3. Reads array
    image = tf.image.decode_image(image, channels=3, dtype=tf.float64)
    
    # add batch dimension
    image = image[tf.newaxis, :]

    return image, label

def read_image_sobel(image_file_path, label):
    """Function to read in images to create tf dataset. Adds an extra dimension for batch. Allows use on sobel filter. """
    
    image = tf.io.read_file(image_file_path)  # read in image
    
    # preserving RGB colors with channels = 3. Reads array
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    
    image, label = augment_crop(image,label)
    
    # add batch dimension
    image = image[tf.newaxis, :]

    return image, label

### Encodings ###
def create_one_hot_encoding(file_paths, labels): 
    
    """convert labels to numeric representation"""
    
    # create encoder
    labelencoder = LabelEncoder()
    
    depth = len(set(labels))

    # apply encoder to change string labels to integer
    labels_encoded = labelencoder.fit_transform(labels)

    # One-hot encodning to feed into model(s)
    labels = tf.one_hot(labels_encoded, depth=depth)

    # create dataset (ds)
    ds = create_tf_ds(file_paths, labels)
    
    return ds, labelencoder


def decoder(ds): 
    """_summary_

    Args:
        ds (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get labels out of tf set
    y = np.concatenate([y for x, y in ds], axis=0) # works since ds is batched
    # labels = labelencoder.inverse_transform(y.argmax(axis = 1))
    labels = y.argmax(axis = 1)
    return labels


def split_ds_tvt(ds, read_image_fn  = read_image, train_size = 0.7, val_size = 0.1, test_size = 0.2): 
    """ Train test split dataset. Read images into tensor dataset.
    
    Args: 
    train_size, val_size, test_size (float): sum to 1
    
    """
    DATASET_SIZE = ds.cardinality().numpy()

    train_size = int(train_size * DATASET_SIZE)
    val_size = int(val_size * DATASET_SIZE)
    test_size = int(test_size * DATASET_SIZE)

    ds = ds.shuffle(buffer_size= DATASET_SIZE)
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    val_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())
    print(tf.data.experimental.cardinality(test_ds).numpy())
    
    # train_ds = train_ds.map(read_image)
    # val_ds = val_ds.map(read_image)
    # test_ds = test_ds.map(read_image)
    
    return train_ds, val_ds, test_ds


def optimize_tensor(ds): 
    """_summary_

    Args:
        ds (_type_): _description_

    Returns:
        _type_: _description_
    """
    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
    return ds

def read_ds(ds, read_fn = read_image): 
    """ Read in images to arrays and augment with predefined process"""
    ds = ds.map(read_fn, num_parallel_calls=AUTOTUNE)
    return ds


def configure_for_performance(ds, batch_size = 32):
    """_summary_

    Args:
        ds (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 32.

    Returns:
        tensorflow dataset: optimized dataset ready for performance in a model
    """
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def preprocess_ds(ds, read_fn = read_image, augment_fn = augment_crop):
    """ Steps to get data ready for model training and testing. 
    
    This is mean to be a way to create a dataset for use in models quickly and effectively. 
    
    Args: 
        ds (tensorflow dataset): created dataset with image, label structure
        read_fn: read image function
    
    Returns: 
        train_ds, val_ds, test_ds (tensor datasets): train, validation, and test datasets for traininga and evaluating models. 
    """
    # split
    train_ds, val_ds, test_ds = split_ds_tvt(ds)
    
    # optimize
    train_ds = optimize_tensor(train_ds)
    val_ds = optimize_tensor(val_ds)
    
    # read in and augment data
    train_ds = read_ds(train_ds, read_fn)
    val_ds = read_ds(val_ds, read_fn)
    test_ds = read_ds(test_ds, read_fn)
    
    return train_ds, val_ds, test_ds
