import os
from typing import Iterable

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorboard.plugins import projector


class Projector:
    def __init__(
            self,
            images: np.array,
            labels: Iterable,
            log_dir: str,
            data_name: str):
        """
        Create all necessary artifacts and configs for Tensorboard Projector.

        Parameters
        ----------
        images : np.array
            An n-d array of images (NumberOFImages X Width X Height).
        labels : Iterable
            A one-to-one asociated labes to images. It can be a list, numpy array, or any iterable.
        log_dir : str
            The location all the artifacts are being saved. The directory to which Tensorboard is directd.
            `Tensorboard --logdir "log_dir"`
        data_name : str
            The name of the dataset, which is appended to the name of all artifacts.
        """
        self.log_dir = log_dir
        self.images = images
        self.labels = labels
        self.data_name = data_name

        self.n_images = None
        self.image_width = None
        self.image_height = None
        self.points = None

        if self.images:
            self.convert_images_to_points()

    def convert_images_to_points(self):
        """
        Convert images array to high-dimentional data points (NumberOFImages X NumberOfDimensions).
        """
        self.n_images, self.image_width, self.image_height = self.images.shape
        self.points = np.reshape(
            self.images, (-1, self.image_width * self.image_height))

    def save_points(self):
        """
        Save high-dimensional data points into a model checkpoint.
        """
        points_filename = os.path.join(self.log_dir, f'images_{self.data_name}.ckpt')
        points_tensor = tf.Variable(self.points, name=self.data_name)
        ckpt = tf.train.Checkpoint(**{self.data_name: points_tensor})
        ckpt.save(points_filename)
        print('> Images are saved in {}'.format(points_filename))

    def save_labels(self):
        """
        Save labels into a metadata tab-separated-value file.
        """
        meta_filename = os.path.join(
            self.log_dir, f'metadata_{self.data_name}.tsv')
        with open(meta_filename, 'w') as metadata_file:
            for row in self.labels:
                metadata_file.write(f'{row}\n')
        print('> Metadata file is saved in {}'.format(meta_filename))

    def write_sprite_image(self):
        """
        Create and write a sprite image, a single PNG file containing all images (possibly downsampled).
        """
        # Calculate number of plot
        n_plots = int(np.ceil(np.sqrt(self.n_images)))

        # Preallocate the sprite image
        sprite_image = np.ones(
            (self.image_height * n_plots, self.image_width * n_plots))

        for i in range(n_plots):
            for j in range(n_plots):
                img_idx = i * n_plots + j
                if img_idx < self.n_images:
                    img = self.images[img_idx]
                    sprite_image[i * self.image_height: (i + 1) * self.image_height,
                                 j * self.image_width: (j + 1) * self.image_width] = img

        sprite_filename = os.path.join(
            self.log_dir, f'sprite_{self.data_name}.png')
        plt.imsave(sprite_filename, sprite_image, cmap='gray')
        print('> Sprite image saved in {}'.format(sprite_filename))

    def create_config(self, with_sprite=True):
        """
        Create a congfig files that defines image tensor name, path to metadata file, path to the sprite image,
        and the size of individual image whithin the sprite image.

        Parameters
        ----------
        with_sprite : bool, optional
            If to save sprite or not, by default True
        """
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = f'{self.data_name}/.ATTRIBUTES/VARIABLE_VALUE'
        embedding.metadata_path = f'metadata_{self.data_name}.tsv'
        if with_sprite:
            embedding.sprite.image_path = f'sprite_{self.data_name}.png'
            embedding.sprite.single_image_dim.extend(
                [self.image_width, self.image_height])
        projector.visualize_embeddings(self.log_dir, config)

    def make(self):
        self.save_points()
        self.save_labels()
        self.write_sprite_image()
        self.create_config()


if __name__ == "__main__":
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    log_dir = '/Users/behrooz/workspace/unsupervised/logs/projector3'
    data_name = 'fmnist_with_image'
    labels = train_labels[:1000]
    images = train_images[:1000]

    proj = Projector(
        images=images,
        labels=labels,
        log_dir=log_dir,
        data_name=data_name)
    proj.make()
