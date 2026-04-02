import os
import cv2
import numpy as np

import pickle
import lz4
import lz4.block
from PIL import Image
from io import BytesIO

from dataset.base_dataset import BaseDataset


class amazon(BaseDataset):
    def __init__(self, data_path, filenames_path='./code/dataset/filenames/',
                is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)

        self.scale_size = scale_size

        self.is_train = is_train

        # Determine which split directory to use
        if is_train:
            split_dir = os.path.join(data_path, 'amazon_data', 'train_data')
        else:
            split_dir = os.path.join(data_path, 'amazon_data', 'test_data')

        # Collect all .pklz files from the split directory
        self.file_paths = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith('.pklz')
        ])

        phase = 'train' if is_train else 'test'
        print("Dataset: Amazon")
        print("# of %s images: %d" % (phase, len(self.file_paths)))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pklz_path = self.file_paths[idx]
        record = self.get_compressed_object(pklz_path)

        # Extract data from the record
        img = record['image_data']       # binary image data
        dims = record['dimensions']      # dimensions as strings in inches

        # Decode image and resize to match ShapeNetSem format
        image = self.unpack_amazon_image(img)
        image = self.resize_image(image)

        # Compute normalization factor from physical dimensions
        dimensions = self.dimensions_to_float(dims)
        normalization = self.getNormalization(dimensions)

        # Build filename from the .pklz filename
        basename = os.path.basename(pklz_path)
        filename = basename.replace('.pklz', '.png')

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
        
        image_tensor = self.to_tensor(image)

        return {'image': image_tensor, 'normalization': normalization, 'image_no_tensor': image, 'filename': filename}
    
    def get_compressed_object(self, filename):
        with open(filename, 'rb') as fp:
            compressed_bytes = fp.read()
        decompressed = lz4.block.decompress(compressed_bytes)
        pickled_object = pickle.loads(decompressed)

        return pickled_object
    
    def unpack_amazon_image(self, binary_image_data):
        bytes_image_data = BytesIO(binary_image_data)
        image = Image.open(bytes_image_data)
        opencv_image = np.array(image)

        return opencv_image
    
    def resize_image(self, image):
        border_size = (640 - 480) // 2
        border_color = [255, 255, 255]
        image = cv2.resize(image, (480, 480))
        image = cv2.copyMakeBorder(image, 0, 0, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

        return image
    
    def dimensions_to_float(self, dimensions):
        return np.array([float(dim) for dim in dimensions])
    
    def get_amazon_test_set(self, path):
        data = self.get_compressed_object(path)
        dimensions = data['dimensions']
        binary_image_data = data['image_data']

        dimensions = self.dimensions_to_float(dimensions)
        image = self.unpack_amazon_image(binary_image_data)
        image = self.resize_image(image)

        return dimensions, image
    
    def getNormalization(self, dimensions):
        dimensions = dimensions * 2.54 # Convert from inches to cm
        dimensions = dimensions / 100.0 # Convert from cm to meters
        return np.linalg.norm(dimensions)
    