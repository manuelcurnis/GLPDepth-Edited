import os
import cv2
import OpenEXR
import Imath
import numpy as np
import pandas as pd

from dataset.base_dataset import BaseDataset


class shapenetsem_normalized(BaseDataset):
    def __init__(self, data_path, filenames_path='./code/dataset/filenames/',
                 metadata_path='./code/dataset/filenames/shapenetsem/metadata.txt', is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'shapenetsem')

        txt_path = os.path.join(filenames_path, 'shapenetsem')

        if is_train:
            txt_path += '/train.txt' #'/test_list.txt' or '/shapenetsem.txt' or '/train.txt'
        else:
            txt_path += '/list_test.txt' #'/test_list.txt' or '/shapenetsem.txt' or '/test.txt' or '/test_small.txt'

        self.filenames_list = self.readTXT(txt_path)
        self.metadata = pd.read_csv(metadata_path)
        phase = 'train' if is_train else 'test'
        print("Dataset: Shapenet Sem (Normalized)")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        if self.is_train:
            return len(self.filenames_list) // 14 * 3
        else:
            return len(self.filenames_list)

    def __getitem__(self, idx):
        if self.is_train:
            idx = idx % (len(self.filenames_list) // 14)
            idx *= 14
            idx += np.random.randint(0, 14, 1)[0]
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-1]

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        alpha = image[:, :, 3]
        image[alpha == 0] = (255, 255, 255, 0)
        image = image[:, :, :3]
        depth = self.read_exr(gt_path)

        normalization = self.getNormalization(img_path)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        return {'image': image, 'depth': depth, 'depth_normalized': depth / normalization, 'normalization': normalization, 'filename': filename}
    
    def read_exr(self, file_path):
        exr_file = OpenEXR.InputFile(file_path)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        redstr = exr_file.channel('R', pt)
        red = np.frombuffer(redstr, dtype = np.float32)
        red.shape = (size[1], size[0]) # Numpy arrays are (row, col)

        return red
    
    def getNormalization(self, img_path):
        id_str = 'wss.' + img_path.split('/')[3]
        index = self.metadata.loc[self.metadata['fullId'] == id_str].index[0]
        aligned_dims = self.metadata['aligned.dims'][index]
        aligned_dims = [float(dim) / 100.0 for dim in aligned_dims.split('\\,')]
        aligned_dims = np.array(aligned_dims)
        return np.linalg.norm(aligned_dims)
