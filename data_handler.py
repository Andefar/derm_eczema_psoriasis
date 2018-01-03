import json
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
from scipy import misc


class DataHandler:

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

        cores = cpu_count()

        parts = self._chunk(local_paths,cores)

        #print("len(parts):",len(parts))

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

    def _get_paths(self):
        out = []
        for subdir, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpe") or file.endswith(".bmp"):
                    out.append(subdir + "/" + file)
        np.random.shuffle(out)
        if len(out) == 0:
            raise("No images found")
        return out


    def __init__(self, img_size, train_frac=.6, val_frac=.2, test_frac=.2, max_images_in_mem_train = 1024, max_images_in_mem_valid = 1024, batch_size = 32 , data_dir='deep_learning'):

        #print("Indexing data...")

        self.H, self.W,self.C = img_size
        self.train_frac, self.val_frac, self.test_frac = train_frac, val_frac, test_frac
        self.batch_size = batch_size

        self.data_path = data_dir
        self.json_dir = 'data_split/'

        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir)

        self.paths = self._get_paths()

        split_train = int(len(self.paths) * self.train_frac)
        split_val = int(len(self.paths) * self.val_frac) + split_train

        self.train_files = self.paths[:split_train]
        self.valid_files = self.paths[split_train:split_val]
        self.test_files = self.paths[split_val:]

        self.paths = []

        #print("train:\t%i\t:\t%i" % (0,split_train))
        #print("valid:\t%i\t:\t%i" % (split_train, split_val))
        #print("test:\t%i\t:" % split_val)

        #print("Dumping filename lists...")
        with open(self.json_dir + 'train_files.json', 'w+') as outfile:
            json.dump(self.train_files,outfile)
        with open(self.json_dir + 'valid_files.json', 'w+') as outfile:
            json.dump(self.valid_files,outfile)
        with open(self.json_dir + 'test_files.json', 'w+') as outfile:
            json.dump(self.test_files,outfile)

        self.train_files = np.array(self.train_files)
        self.valid_files = np.array(self.valid_files)
        self.test_files = np.array(self.test_files)

        self.paths_train = []
        self.paths_valid = []

        self.train_size = len(self.train_files)
        self.valid_size = len(self.valid_files)

        if not os.path.exists(self.json_dir ):
            os.makedirs(self.json_dir )

        self.X_batch_train = []
        self.y_batch_train = []

        self.X_batch_valid = []
        self.y_batch_valid = []

        self.total_num_batches_train = int(np.ceil(self.train_size / float(batch_size)))  # 120 / 10 = 12 -> 12 batches in total
        self.batch_per_mem_chunk_train = int(np.floor(float(max_images_in_mem_train) / float(batch_size))) # 64 / 10 = 6 train batches per chunk

        self.total_num_batches_valid = int(np.ceil(self.valid_size / float(batch_size)))  # 40 / 10 = 4 -> 4 batches in total
        self.batch_per_mem_chunk_valid = int(np.floor(float(max_images_in_mem_valid) / float(batch_size))) # 64 / 10 = 6 batches per chunk

        self.current_mem_chunk_idx_train = 0
        self.cur_start_idx_train = 0
        self.cur_end_idx_train = self.batch_size

        self.current_mem_chunk_idx_valid = 0
        self.cur_start_idx_valid = 0
        self.cur_end_idx_valid = self.batch_size



    def shuffle_and_index_train_data(self):

        #print("Shuffeling data...")

        rand_indices_train = np.random.choice(self.train_size, self.train_size, replace=False)
        rand_indices_valid = np.random.choice(self.valid_size, self.valid_size, replace=False)

        #print("rand_indices_train shape: ", rand_indices_train.shape)
        #print("rand_indices_valid shape: ", rand_indices_valid.shape)

        self.paths_train = []
        for batch_idx in range(self.total_num_batches_train):
            if batch_idx != self.total_num_batches_train - 1:
                self.paths_train.append(rand_indices_train[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size])
            else:
                self.paths_train.append(rand_indices_train[batch_idx * self.batch_size : ] )

        #print("self.paths_train: ",self.paths_train)

        self.paths_valid = []
        for batch_idx in range(self.total_num_batches_valid):
            if batch_idx != self.total_num_batches_valid - 1:
                self.paths_valid.append(rand_indices_valid[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size])
            else:
                self.paths_valid.append(rand_indices_valid[batch_idx * self.batch_size : ])

        #print("self.paths_valid: ", self.paths_valid)

        self.current_mem_chunk_idx_train = 0
        self.current_mem_chunk_idx_valid = 0


    def load_train_batch(self,i):

        if i == self.current_mem_chunk_idx_train:
            #print("Getting new train chunk...")
            chunk_img_filenames = []

            start = i
            end = i + self.batch_per_mem_chunk_train if (i + self.batch_per_mem_chunk_train) <= (self.total_num_batches_train - 1) else self.total_num_batches_train

            #print("train - range(%i,%i,1) = %s" % (start,end,str(list(range(start,end, 1)))))
            for batch_idx in range(start,end, 1):

                #print(type(self.train_files[[self.paths[batch_idx]]]))
                chunk_img_filenames.extend(self.train_files[[self.paths_train[batch_idx]]])
                #chunk_img_filenames = np.concatenate((chunk_img_filenames, self.train_files[[self.paths[batch_idx]]]))


            #print("Loading %i train images" % len(chunk_img_filenames))
            #print("Train - chunk_img_filenames:\n",chunk_img_filenames)
            self.X_batch_train, self.y_batch_train = self._get_data_split(np.array(chunk_img_filenames))

            self.current_mem_chunk_idx_train = i + self.batch_per_mem_chunk_train

        max_samples_train = len(self.X_batch_train)

        if i < self.total_num_batches_train - 1:

            #print("Train - Returning [%i : %i]" % (self.cur_start_idx_train, self.cur_end_idx_train))
            X_temp, y_temp = np.array(self.X_batch_train[self.cur_start_idx_train : self.cur_end_idx_train]), np.array(self.y_batch_train[self.cur_start_idx_train : self.cur_end_idx_train])
            #print("Updating training indices")
            self._update_indices_train(max_samples_train)
            #print("Returning training images")
            return X_temp, y_temp

        else:

            #print("Train - Returning [%i : ] \t (end)" % (self.cur_start_idx_train))
            #print("Updating trainging indices")
            X_temp, y_temp = np.array(self.X_batch_train[self.cur_start_idx_train : ]), np.array(self.y_batch_train[self.cur_start_idx_train : ])
            self._update_indices_train(max_samples_train)
            #print("Returning training images")
            return X_temp, y_temp

    def load_valid_batch(self,i):

        if i == self.current_mem_chunk_idx_valid:
            #print("Getting new valid chunk...")
            chunk_img_filenames = []

            start = i
            end = i + self.batch_per_mem_chunk_valid if (i + self.batch_per_mem_chunk_valid) <= (self.total_num_batches_valid - 1) else self.total_num_batches_valid

            #print("valid - range(%i,%i,1) = %s" % (start,end,str(list(range(start,end, 1)))))
            for batch_idx in range(start,end, 1):

                #print(type(self.train_files[[self.paths[batch_idx]]]))
                chunk_img_filenames.extend(self.valid_files[[self.paths_valid[batch_idx]]])
                #chunk_img_filenames = np.concatenate((chunk_img_filenames, self.train_files[[self.paths[batch_idx]]]))


            #print("Loading %i validation images" % len(chunk_img_filenames))
            #print("Valid chunk_img_filenames:\n",chunk_img_filenames)
            self.X_batch_valid, self.y_batch_valid = self._get_data_split(np.array(chunk_img_filenames))

            self.current_mem_chunk_idx_valid = i + self.batch_per_mem_chunk_valid

        max_samples_valid = len(self.X_batch_valid)

        if i < self.total_num_batches_valid - 1:

            #print("Valid - Returning [%i : %i]" % (self.cur_start_idx_valid, self.cur_end_idx_valid))
            X_temp_valid, y_temp_valid = np.array(self.X_batch_valid[self.cur_start_idx_valid : self.cur_end_idx_valid]), np.array(self.y_batch_valid[self.cur_start_idx_valid : self.cur_end_idx_valid])
            #print("Updating validation indices")
            self._update_indices_valid(max_samples_valid)
            #print("Returning validation images")
            return X_temp_valid, y_temp_valid

        else:

            #print("Valid - Returning [%i : ] \t (end)" % (self.cur_start_idx_valid))
            X_temp_valid, y_temp_valid = np.array(self.X_batch_valid[self.cur_start_idx_valid : ]), np.array(self.y_batch_valid[self.cur_start_idx_valid : ])
            #print("Updating validation indices")
            self._update_indices_valid(max_samples_valid)
            #print("Returning validation images")
            return X_temp_valid, y_temp_valid


    def _update_indices_train(self,max_samples_train):
        self.cur_start_idx_train += self.batch_size
        self.cur_end_idx_train += self.batch_size

        if self.cur_start_idx_train >= max_samples_train:
            self.cur_start_idx_train = 0

        if self.cur_end_idx_train >= max_samples_train + self.batch_size:
            self.cur_end_idx_train = self.batch_size


    def _update_indices_valid(self,max_samples_valid):
        self.cur_start_idx_valid += self.batch_size
        self.cur_end_idx_valid += self.batch_size

        if self.cur_start_idx_valid >= max_samples_valid:
            self.cur_start_idx_valid = 0

        if self.cur_end_idx_valid >= max_samples_valid + self.batch_size:
            self.cur_end_idx_valid = self.batch_size


    def load_test_data(self):
        self.X_test, self.y_test = self._get_data_split(self.test_files)
        return np.array(self.X_test), np.array(self.y_test)


    def get_train_size(self):
        return self.train_size


    def get_valid_size(self):
        return self.valid_size


