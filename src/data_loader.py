# Use `pip install [library name]` in the conda environement to install these libraries

import tensorflow as tf
import numpy as np
import os
from tensorflow import data
import nltk
from PIL import Image
from build_vocab import Vocabulary
import random
# import lmdb
import pickle


class EpicuriousDataset(data.Dataset):

    def __init__(self, data_dir, aux_data_dir, split, maxseqlen, maxnuminstrs, maxnumlabels, maxnumims,
                 transform=None, max_num_samples=-1, use_lmdb=False, suff=''):
        
        """
        Initializes the EpicuriousDataset class.

        Args:
            data_dir (str): Directory containing image data.
            aux_data_dir (str): Directory containing auxiliary data.
            split (str): Dataset split (train/val/test).
            maxseqlen (int): Maximum sequence length.
            maxnuminstrs (int): Maximum number of instructions.
            maxnumlabels (int): Maximum number of labels.
            maxnumims (int): Maximum number of images.
            transform (Tensorflow layers): Image transformation layers.
            max_num_samples (int): Maximum number of samples to load (-1 means load all).
            use_lmdb (bool): Whether to use LMDB for image loading.
            suff (str): Suffix for auxiliary data files.
        """

        ## Load data from aux_data_dir. Use suff to help with this.
        self.ingrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'epicurious_vocab_ingrs.pkl'), 'rb'))
        self.instrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'epicurious_vocab_toks.pkl'), 'rb'))
        self.dataset = pickle.load(open(os.path.join(aux_data_dir, suff + 'epicurious_'+split+'.pkl'), 'rb'))

        self.label2word = self.get_ingrs_vocab()

        # If a lambda function is provided, ensure that it is run correctly.
        # self.use_lmdb = use_lmdb
        # if use_lmdb:
        #     self.image_file = lmdb.open(os.path.join(aux_data_dir, 'lmdb_' + split), max_readers=1, readonly=True,
        #                                 lock=False, readahead=False, meminit=False)

        # Keep only the entries in the dataset that contain an image.
        self.ids = []
        self.split = split
        for i, entry in enumerate(self.dataset):
            if len(entry['images']) == 0:
                continue
            self.ids.append(i)

        # Check that these instance variables are initialized properly.
        self.root = os.path.join(data_dir, 'images')
        self.transform = transform
        self.max_num_labels = maxnumlabels
        self.maxseqlen = maxseqlen
        self.max_num_instrs = maxnuminstrs
        self.maxseqlen = maxseqlen*maxnuminstrs
        self.maxnumims = maxnumims
        if max_num_samples != -1:
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples]

    def get_instrs_vocab(self):
        return self.instrs_vocab

    def get_instrs_vocab_size(self):
        return len(self.instrs_vocab)

    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient

    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        # Get the sample from the dataset using the index
        sample = self.dataset[self.ids[index]]
        img_id = sample['id']
        captions = sample['tokenized']
        paths = sample['images'][0:self.maxnumims]
        idx = index

        valid_paths = [path for path in paths if '#NAME?' not in path]
        if not valid_paths:
            return None  

        # Get the labels (ingredients) and title of the recipe
        labels = self.dataset[self.ids[idx]]['ingredients']
        title = sample['title']

        # Initialize tokens with the title and a separator
        tokens = []
        tokens.extend(title)
        tokens.append('<eoi>') # add fake token to separate title from recipe

        # Extend tokens with the captions and a separator
        for c in captions:
            tokens.extend(c)
            tokens.append('<eoi>')

        # Initialize ground truth labels with padding
        ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
        pos = 0

        # Convert ingredient labels to indices
        true_ingr_idxs = []
        for i in range(len(labels)):
            true_ingr_idxs.append(self.ingrs_vocab(labels[i]))

        # Populate ground truth labels with ingredient indices
        for i in range(self.max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_idx = self.ingrs_vocab(label)
            if label_idx not in ilabels_gt:
                ilabels_gt[pos] = label_idx
                pos += 1

        # Mark the end of the ground truth labels
        ilabels_gt[pos] = self.ingrs_vocab('<end>')
        ingrs_gt = tf.cast(tf.convert_to_tensor(ilabels_gt), dtype=tf.int64)

        if len(paths) == 0:
            path = None
            image_input = tf.zeros((3, 224, 224))
        else:
            if self.split == 'train':
                img_idx = np.random.randint(0, len(paths))
            else:
                img_idx = 0
            path = paths[img_idx] + ".jpg"
            # if self.use_lmdb:
            #     try:
            #         with self.image_file.begin(write=False) as txn:
            #             image = txn.get(path.encode())
            #             image = np.fromstring(image, dtype=np.uint8)
            #             image = np.reshape(image, (256, 256, 3))
            #         image = Image.fromarray(image.astype('uint8'), 'RGB')
            #     except:
            #         print ("Image id not found in lmdb. Loading jpeg file...")
            #         image = Image.open(os.path.join(self.root, path[0], path[1],
            #                                         path[2], path[3], path)).convert('RGB')
            # else:
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            # if len(self.transform) != 0:
            #     for func in enumerate(self.transform):
            #         image = func(image)

            # NOTE: Convert image to a Tensor
            image = tf.convert_to_tensor(tf.keras.utils.img_to_array(image)) # defaults to float32 Tensor

            if self.transform is not None:
                image = self.transform(image)
                
            image_input = image

        # Convert caption (string) to word ids.
        caption = []

        caption = self.caption_to_idxs(tokens, caption)
        caption.append(self.instrs_vocab('<end>'))

        caption = caption[0:self.maxseqlen]
        # target = tf.Tensor(caption)
        target = tf.convert_to_tensor(caption)

        return image_input, target, ingrs_gt, img_id, path, self.instrs_vocab('<pad>')

    def __len__(self):
        return len(self.ids)

    def caption_to_idxs(self, tokens, caption):
        caption.append(self.instrs_vocab('<start>'))
        for token in tokens:
            caption.append(self.instrs_vocab(token))
        return caption
    
    def _inputs(self):
        pass

    def element_spec(self):
        pass

class DataLoader():
    def __init__(self, dataset, batch_size, shuffle, num_workers, drop_last, collate_fn, pin_memory):
        '''
        dataset: data.Dataset - the dataset to sample over
        batch_size: int - batch size
        shuffle: Boolean - whether to shuffle the data between epochs
        num_workers: int - the number of subprocs (?)
        drop_last: Boolean - whether to drop the last non-complete batch
        collate_fn: Callable - callable to collate the data
        pin_memory: Boolean
        '''
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.index = 0

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        else:
            self.indices = np.arange(len(self.dataset))
        return self

    # def __next__(self):
    #     if self.index >= len(self.dataset):
    #         raise StopIteration
    #     start = self.index
    #     self.index += self.batch_size
    #     if self.index > len(self.dataset):
    #         if self.drop_last:
    #             raise StopIteration
    #         self.index = len(self.dataset)
    #     batch_indices = self.indices[start:self.index]
    #     batch = [self.dataset[i] for i in batch_indices]
        
    #     if self.collate_fn:
    #         batch = self.collate_fn(batch)
    #     return batch

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        
        batch = []
        while len(batch) < self.batch_size:
            if self.index >= len(self.dataset):
                if self.drop_last or not batch:
                    raise StopIteration
                else:
                    break  # Use whatever is already collected in the batch
            # Get the index of the next item
            idx = self.indices[self.index]
            self.index += 1
            
            # Fetch the data item from the dataset
            data_item = self.dataset[idx]
            if data_item is not None:
                batch.append(data_item)

        if self.collate_fn and batch:
            return self.collate_fn(batch)
        return batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# def collate_fn(data):

#     # Sort a data list by caption length (descending order).
#     # data.sort(key=lambda x: len(x[2]), reverse=True)
#     image_input, captions, ingrs_gt, img_id, path, pad_value = zip(*data)

#     # Merge images (from tuple of 3D tensor to 4D tensor).

#     image_input = tf.stack(image_input, 0)
#     ingrs_gt = tf.stack(ingrs_gt, 0)

#     # Merge captions (from tuple of 1D tensor to 2D tensor).
#     lengths = [len(cap) for cap in captions]
#     targets = tf.cast(tf.ones([len(captions), max(lengths)], dtype=tf.int64), dtype=tf.int64) * pad_value[0]

#     for i, cap in enumerate(captions):
#         end = lengths[i]
#         targets[i, :end] = cap[:end]

#     return image_input, targets, ingrs_gt, img_id, path
def collate_fn(data):
    image_input, captions, ingrs_gt, img_id, path, pad_value = zip(*data)

    image_input = tf.stack(image_input, 0)
    ingrs_gt = tf.stack(ingrs_gt, 0)


    padded_captions = []
    max_length = max(len(cap) for cap in captions)

    for cap in captions:
        padding_size = max_length - len(cap)
        padding = tf.fill([padding_size], pad_value[0])
        padded_caption = tf.concat([cap, padding], axis=0)
        padded_captions.append(padded_caption)

    padded_captions = tf.stack(padded_captions)

    return image_input, padded_captions, ingrs_gt, img_id, path



def get_loader(data_dir, aux_data_dir, split, maxseqlen,
               maxnuminstrs, maxnumlabels, maxnumims, transform, batch_size,
               shuffle, num_workers, drop_last=False,
               max_num_samples=-1,
               use_lmdb=False,
               suff=''):

    dataset = EpicuriousDataset(data_dir=data_dir, aux_data_dir=aux_data_dir, split=split,
                              maxseqlen=maxseqlen, maxnumlabels=maxnumlabels, maxnuminstrs=maxnuminstrs,
                              maxnumims=maxnumims,
                              transform=transform,
                              max_num_samples=max_num_samples,
                              use_lmdb=use_lmdb,
                              suff=suff)

    data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
    return data_loader, dataset
