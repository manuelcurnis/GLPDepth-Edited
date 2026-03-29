import os
import cv2
import OpenEXR
import Imath
import numpy as np

from dataset.base_dataset import BaseDataset


class shapenetsem(BaseDataset):
    def __init__(self, data_path, filenames_path='./code/dataset/filenames/',
                 is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'shapenetsem')

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = os.path.join(filenames_path, 'shapenetsem')

        if is_train:
            txt_path += '/train.txt' #'/test_list.txt' or '/shapenetsem.txt' or '/train.txt'
        else:
            txt_path += '/list_test.txt' #'/test_list.txt' or '/shapenetsem.txt' or '/test.txt'

        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'
        print("Dataset: Shapenet Sem")
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
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = self.read_exr(gt_path)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        return {'image': image, 'depth': depth, 'filename': filename}
    
    def read_exr(self, file_path):
        exr_file = OpenEXR.InputFile(file_path)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        redstr = exr_file.channel('R', pt)
        red = np.frombuffer(redstr, dtype = np.float32)
        red.shape = (size[1], size[0]) # Numpy arrays are (row, col)

        return red
