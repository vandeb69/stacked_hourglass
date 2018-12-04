import os
import numpy as np
import random
from math import ceil
import cv2
import time
from collections import defaultdict
import scipy.misc as scm
import pickle

from utils.dirs import ensure_dir

IMG_DIR = "data/wheat"
TRAIN_DATA_FILE = IMG_DIR + '/training_txt_file.txt'
LOADER_DIR = "loaders/run4"


class StackedHourglassLoaderClass:
    def __init__(self, config):
        pass

    def initialize(self, config):
        self.loader_dir = os.path.join(config.loader_dir, config.name)
        ensure_dir(self.loader_dir)

        self.img_dir = config.img_dir
        self.train_data_file = config.data_file

        self.classes_list = ['spikelets', 'earbases', 'eartips']

        self.batch_size = config.batch_size

        self.select_list = config.get("categories")
        if self.select_list is None:
            self.select_list = self.classes_list
        elif not all([el in self.classes_list for el in self.select_list]):
            raise ValueError("Argument 'select' needs to contain one or more elements out of {}".format(self.classes_list))

        self.n_stacks = config.n_stacks
        self.norm = config.norm
        self.rand = config.rand

        self.train_table = []
        self.data_dict = defaultdict()
        self.valid_set = []
        self.train_set = []

        self.generate_set(rand=self.rand)

    def _create_train_table(self):

        input_file = open(self.train_data_file, 'r')

        print('READING TRAIN DATA')

        for line in input_file:
            line = line.strip()
            line = line.split(' ')
            name = line[0]
            cat = self.classes_list[0]
            self.data_dict[name] = defaultdict()
            for x in line[1:]:
                if x in self.classes_list:
                    cat = x
                    self.data_dict[name][cat] = []
                else:
                    x = int(float(x))
                    self.data_dict[name][cat].append(x)
            for cat in self.classes_list:
                self.data_dict[name][cat] = np.reshape(self.data_dict[name][cat], (-1, 2))

            self.train_table.append(name)

        input_file.close()

    def _randomize(self):
        random.shuffle(self.train_table)

    def _create_sets(self, valid_rate=0.1):
        print('START SET CREATION')
        sample = len(self.train_table)
        valid_sample = int(sample * valid_rate)
        self.train_set = self.train_table[:sample - valid_sample]
        self.valid_set = self.train_table[sample - valid_sample:]
        print('SET CREATED')
        np.save(os.path.join(self.loader_dir, 'Dataset-Validation-Set'), self.valid_set)
        np.save(os.path.join(self.loader_dir, 'Dataset-Training-Set'), self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

    def generate_set(self, rand=False):
        self._create_train_table()
        if rand:
            self._randomize()
        self._create_sets()

    def open_img(self, name, color='RGB'):
        img = cv2.imread(os.path.join(self.img_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR.')

    @staticmethod
    def _crop_box(height, width, center, crop_size=384):
        box = [center[0] - crop_size//2, center[1] - crop_size//2,
               center[0] + crop_size//2, center[1] + crop_size//2]
        if box[0] < 0:
            val = abs(box[0])
            box[0] += val
            box[2] += val
        if box[1] < 0:
            val = abs(box[1])
            box[1] += val
            box[3] += val
        if box[2] > width - 1:
            val = box[2] - width
            box[0] -= val
            box[2] -= val
        if box[3] > height - 1:
            val = box[3] - height
            box[1] -= val
            box[3] -= val

        return box

    @staticmethod
    def _crop_img(img, cbox):
        img = img[cbox[1]:cbox[3], cbox[0]:cbox[2]]
        return img

    def _relative_points(self, box, points, to_size=64):
        new_points = {key: points[key] for key in self.select_list}
        box = np.array(box)
        b_size = max(box[2:] - box[:2])
        for cat in self.select_list:
            new_points[cat] = points[cat] - box[:2]
            new_points[cat] = new_points[cat] * (to_size / b_size)
            new_points[cat].astype(np.int32)
        return new_points

    @staticmethod
    def _make_gaussian(height, width, sigma=1, centers=None):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)
        x0 = centers[:, 0].reshape((-1, 1))
        y0 = centers[:, 1].reshape((-1, 1))
        gx = np.exp(-4 * np.log(2) * ((x - x0)**2 / sigma**2))
        gy = np.exp(-4 * np.log(2) * ((y - y0)**2 / sigma**2))
        g = np.einsum('ij,ik->ijk', gy, gx)
        return np.sum(g, axis=0)

    def _generate_hm(self, height, width, points, sigmas=None):
        if sigmas is None:
            sigmas = {'earbases': 1.5, 'eartips': 1.5, 'spikelets': 0.7}
        num_classes = len(self.select_list)
        hm = np.zeros((height, width, num_classes), dtype=np.float32)

        for i in range(num_classes):
            cat = self.select_list[i]
            centers = points[cat]
            hm[:, :, i] = self._make_gaussian(height, width, sigma=sigmas[cat], centers=centers)

        return hm

    def _get_random_ear_center(self, name):
        ear_tips = self.data_dict[name]['eartips']
        ear_bases = self.data_dict[name]['earbases']
        avg_ear_length = np.mean(np.linalg.norm(ear_bases - ear_tips, axis=1))
        ear_centers = list(map(lambda x: np.mean(np.array(x), axis=0).astype(int),
                               zip(ear_tips, ear_bases)))
        return random.choice(ear_centers), avg_ear_length

    def _generator(self, batch_size, nstacks=4, set='train', normalize=True):
        if set == 'train':
            img_names = self.train_set
        elif set == "valid":
            img_names = self.valid_set
            batch_size = len(img_names)
        n = len(img_names)

        i = 0
        k = 0
        while i < n:
            if batch_size != n and k == (ceil(n / batch_size) - 1):
                batch_size = n % batch_size
                if batch_size == 0:
                    break
            train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, nstacks, 64, 64, len(self.select_list)), np.float32)
            train_avg_earlengths = np.zeros(batch_size, np.float32)
            j = 0
            while j < batch_size:
                if i == n:
                    break
                name = img_names[i]
                img = self.open_img(name)

                center, avg_length = self._get_random_ear_center(name)
                train_avg_earlengths[j] = avg_length * (64 / 512)

                cbox = self._crop_box(img.shape[0], img.shape[1], center, crop_size=512)
                new_p = self._relative_points(cbox, self.data_dict[name], to_size=64)
                hm = self._generate_hm(64, 64, new_p, sigmas=None)

                img = self._crop_img(img, cbox)
                img = scm.imresize(img, (256, 256))
                hm = np.expand_dims(hm, axis=0)
                hm = np.repeat(hm, nstacks, axis=0)
                if normalize:
                    train_img[j] = img.astype(np.float32) / 255
                else:
                    train_img[j] = img.astype(np.float32)
                train_gtmap[j] = hm

                j += 1
                i += 1
            k += 1
            yield train_img, train_gtmap, train_avg_earlengths

    def generator(self, batch_size, set='train'):
        return self._generator(batch_size=batch_size, nstacks=self.n_stacks, normalize=self.norm, set=set)

    def test(self, to_wait=0.2):
        self._create_train_table()
        self._create_sets()

        for i in range(len(self.train_set)):
            img = self.open_img(self.train_set[i])
            center, _ = self._get_random_ear_center(self.train_set[i])
            box = self._crop_box(img.shape[0], img.shape[1], center, crop_size=512)
            new_p = self._relative_points(box, self.data_dict[self.train_set[i]], to_size=256)
            rhm = self._generate_hm(256, 256, new_p, sigmas={'earbases': 4.0, 'eartips': 4.0, 'spikelets': 2.8})
            rimg = self._crop_img(img, box)
            rimg = scm.imresize(rimg, (256, 256))
            # grimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
            cv2.imshow('image', rimg / 255 + rhm)
            if i > 0:
                time.sleep(to_wait)
            if cv2.waitKey(1) == 27:
                print('Ended')
                cv2.destroyAllWindows()
                break

    def save(self):
        fn = os.path.join(self.loader_dir, "data_loader.pkl")
        with open(fn, "wb") as f:
            pickle.dump(self, f, protocol=0)
        print(f"Data loader saved in {fn}.")


class StackedHourglassLoader(StackedHourglassLoaderClass):
    def __new__(cls, config):
        ensure_dir(os.path.join(config.loader_dir, config.name))
        for file in os.listdir(os.path.join(config.loader_dir, config.name)):
            if file.endswith('pkl'):
                fn = os.path.join(config.loader_dir, config.name, file)
                with open(fn, 'rb') as f:
                    obj = pickle.load(f)
                print(f"Data loader loaded from {fn}.")
                return obj
        else:
            obj = super(StackedHourglassLoader, cls).__new__(cls)
            obj.initialize(config)
            obj.save()
            return obj


if __name__ == "__main__":
    from easydict import EasyDict

    config = EasyDict({"img_dir": "data/wheat",
                       "data_file": "data/wheat/training_txt_file.txt",
                       "loader_dir": "loaders/run4",
                       "logs_dir": "/tmp/tensorflow_logs",
                       "categories": ["eartips"],
                       "batch_size": 100,
                       "n_stacks": 1,
                       "norm": True,
                       "rand": False})

    loader = StackedHourglassLoader(config)
    # loader.test(.5)

    generator = loader.generator(batch_size=config.batch_size, set='valid')
    while True:
        img, gtmap, l = next(generator)
        print(img.shape)
