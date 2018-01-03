import os,re
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
from scipy import misc
import json


class DataHandler2:

    def _get_all_pics(self,id,filenames):
        out = {}
        for filename in filenames:
            img = misc.imread(filename)
            if (img.shape == ()):
                raise("File could not be loaded: ",filename," with shape: ", img.shape)
            img = misc.imresize(img, (self.H, self.W)) / 255
            out[filename] = img.reshape((self.H, self.W, self.C))
        return out

    def _chunk(self,seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(np.array(seq[int(last):int(last + avg)]))
            last += avg

        return np.array(out)

    def _get_data_split(self,local_paths):

        cores = cpu_count() * 2

        parts = self._chunk(local_paths,cores)

        args = []
        for i in range(cores):
            inner = []
            inner.append(i)
            inner.append(parts[i])
            args.append(inner)

        with Pool(processes=cores) as pool2:
            output = pool2.starmap(self._get_all_pics, args)

            pics = []
            labels =[]
            for d in output:
                for filename, picture in d.items():
                    pics.append(picture)
                    labels.append(0 if "eczema" in filename else 1)

            return np.array(pics),np.array(labels)

    def _get_paths(self,data_path):
        out = []
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpe") or file.endswith(".bmp"):
                    out.append(subdir + "/" + file)
        np.random.shuffle(out)
        if len(out) == 0:
            raise("No images found")
        return out


    def __init__(self, img_size, train_frac=.6, val_frac=.2, test_frac=.2, data_dir='deep_learning',mode='train',test_class='eczema'):

        paths = self._get_paths(data_dir)

        self.H, self.W, self.C = img_size

        if mode == 'test':
            test_files = json.load(open('json_data/test_files.json'))
            #print(test_files)
            test_files = [v for v in test_files if test_class in v]
            # print(test_files)
            self.X_test, self.y_test = self._get_data_split(test_files)
            #print(self.X_test,'\n', self.y_test)

        else:

            split_all = int(len(paths) * (train_frac + val_frac))
            split_tv = int(len(paths) * (train_frac))
            all_files = paths[:split_all]
            self.test_files = paths[split_all:]

            self.json_dir = "json_data/"

            if not os.path.exists(self.json_dir):
                os.makedirs(self.json_dir)

            with open(self.json_dir + 'all_files.json', 'w+') as outfile:
                json.dump(all_files,outfile)
            with open(self.json_dir + 'test_files.json', 'w+') as outfile:
                json.dump(self.test_files,outfile)


            X, y = self._get_data_split(all_files)

            self.X_train, self.y_train = np.array(X[:split_tv]), np.array(y[:split_tv])
            self.X_valid, self.y_valid = np.array(X[split_tv:]), np.array(y[split_tv:])


    def load_data(self):
        return self.X_train, self.y_train, self.X_valid, self.y_valid

    def load_test_data(self):
        return self.X_test, self.y_test


if __name__ == '__main__':

    data_dir = 'data_test/'
    H, W, C = 224, 224, 3

    dh = DataHandler2((H, W, C), data_dir=data_dir)
