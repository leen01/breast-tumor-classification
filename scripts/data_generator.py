from imgaug import augmenters as iaa
import numpy as np
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

# Data generator
class breakHis_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, shape, shuffle=False, use_cache=False, augment=False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        
        if use_cache == True:
            self.cache = np.zeros(
                (paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
            
        self.on_epoch_end()

    def __len__(self):
        """ Length of object"""
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx *
                               self.batch_size: (idx+1) * self.batch_size]

        paths = np.array(self.paths)[indexes]
        X = np.zeros((paths.shape[0], self.shape[0],
                     self.shape[1], self.shape[2]))
        
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = np.array(self.labels)[indexes]

        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),  # horizontal flips
                    iaa.Flipud(0.5),

                    iaa.ContrastNormalization((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(
                        0.0, 0.05*255), per_channel=0.5),
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    iaa.Affine(rotate=0)                    
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(
                X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        image = tf.io.read_file(path)  # read in image

        # preserving RGB colors with channels = 3. Reads array
        image = tf.image.decode_image(image, channels=3, dtype=tf.float64)

        # normalize image size to a square and get smaller subsection
        image = tf.image.random_crop(value=image, size=(224, 224, 3))

        # im = resize(image_norm, (shape[0], shape[1],shape[2]), mode='reflect')

        return image