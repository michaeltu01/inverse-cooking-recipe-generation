import nltk
import pickle
import argparse
from collections import Counter
import json
import os
from tqdm import *
import numpy as np
import re
import pandas as pd
import ast

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word, idx=None):
        if idx is None:
            if not word in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
            return self.idx
        else:
            if not word in self.word2idx:
                self.word2idx[word] = idx
                if idx in self.idx2word.keys():
                    self.idx2word[idx].append(word)
                else:
                    self.idx2word[idx] = [word]

                return idx

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<pad>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_ingredient(det_ingr, replace_dict):
    det_ingr_undrs = det_ingr['text'].lower()
    det_ingr_undrs = ''.join(i for i in det_ingr_undrs if not i.isdigit())

    for rep, char_list in replace_dict.items():
        for c_ in char_list:
            if c_ in det_ingr_undrs:
                det_ingr_undrs = det_ingr_undrs.replace(c_, rep)
    det_ingr_undrs = det_ingr_undrs.strip()
    det_ingr_undrs = det_ingr_undrs.replace(' ', '_')

    return det_ingr_undrs


def get_instruction(instruction, replace_dict, instruction_mode=True):
    instruction = instruction.lower()

    for rep, char_list in replace_dict.items():
        for c_ in char_list:
            if c_ in instruction:
                instruction = instruction.replace(c_, rep)
        instruction = instruction.strip()
    # remove sentences starting with "1.", "2.", ... from the targets
    if len(instruction) > 0 and instruction[0].isdigit() and instruction_mode:
        instruction = ''
    return instruction


def remove_plurals(counter_ingrs, ingr_clusters):
    del_ingrs = []

    for k, v in counter_ingrs.items():

        if len(k) == 0:
            del_ingrs.append(k)
            continue

        gotit = 0
        if k[-2:] == 'es':
            if k[:-2] in counter_ingrs.keys():
                counter_ingrs[k[:-2]] += v
                ingr_clusters[k[:-2]].extend(ingr_clusters[k])
                del_ingrs.append(k)
                gotit = 1

        if k[-1] == 's' and gotit == 0:
            if k[:-1] in counter_ingrs.keys():
                counter_ingrs[k[:-1]] += v
                ingr_clusters[k[:-1]].extend(ingr_clusters[k])
                del_ingrs.append(k)
    for item in del_ingrs:
        del counter_ingrs[item]
        del ingr_clusters[item]
    return counter_ingrs, ingr_clusters


def cluster_ingredients(counter_ingrs):
    '''
    Cluster similar ingredients (e.g. oil, cheese, pasta).
    '''
    mydict = dict()
    mydict_ingrs = dict()

    for k, v in counter_ingrs.items():

        w1 = k.split('_')[-1]
        w2 = k.split('_')[0]
        lw = [w1, w2]
        if len(k.split('_')) > 1:
            w3 = k.split('_')[0] + '_' + k.split('_')[1]
            w4 = k.split('_')[-2] + '_' + k.split('_')[-1]

            lw = [w1, w2, w4, w3]

        gotit = 0
        for w in lw:
            if w in counter_ingrs.keys():
                # check if its parts are
                parts = w.split('_')
                if len(parts) > 0:
                    if parts[0] in counter_ingrs.keys():
                        w = parts[0]
                    elif parts[1] in counter_ingrs.keys():
                        w = parts[1]
                if w in mydict.keys():
                    mydict[w] += v
                    mydict_ingrs[w].append(k)
                else:
                    mydict[w] = v
                    mydict_ingrs[w] = [k]
                gotit = 1
                break
        if gotit == 0:
            mydict[k] = v
            mydict_ingrs[k] = [k]

    return mydict, mydict_ingrs


def build_vocab_epicurious(args):
    print ("Loading data...")

    ## Load data from CSV into pickle files
    CSV_PATH = os.path.join(args.epicurious_path, 'epicurious_data.csv')
    COLS = ['ID','Title','Ingredients','Instructions','Image_Name','Cleaned_Ingredients'] # not using the index or uncleaned Ingredients column in CSV
    DTYPES = {
        'ID': 'int',
        'Title': 'str',
        'Ingredients': 'str',
        'Instructions': 'str',
        'Image_Name': 'str',
        'Cleaned_Ingredients': 'str'
    }

    # Read the CSV into a pandas dataframe for ease of manipulation
    dataset_df = pd.read_csv(
        filepath_or_buffer=CSV_PATH,
        header=0, # marks the first row as the header row
        usecols=COLS,
        dtype=DTYPES
    )

    # Convert to pkl file and save it
    dataset_df.to_pickle(os.path.join(args.save_path, 'epicurious_dataset.pkl'))

    print("Loaded data.")
    print(f"Loaded {dataset_df.size} recipes from the Epicurious Dataset.")

    id2im = {}
    ingrs_file = args.save_path + 'allingrs_count.pkl'
    instrs_file = args.save_path + 'allwords_count.pkl'

    #####
    # 1. Count words in dataset and clean
    #####
    if os.path.exists(ingrs_file) and os.path.exists(instrs_file) and not args.forcegen:
        print ("loading pre-extracted word counters")
        counter_ingrs = pickle.load(open(args.save_path + 'allingrs_count.pkl', 'rb'))
        counter_toks = pickle.load(open(args.save_path + 'allwords_count.pkl', 'rb'))
    else:
        counter_ingrs = Counter()
        counter_toks = Counter()

        for i, row in dataset_df.iterrows():
            # get the instructions for this recipe
            instrs: str = row['Instructions']
            ingrs: str = row['Cleaned_Ingredients']

            # split the recipe into a list of instructions (list of words)
            acc_len = 0 # cumulative num of words
            instrs_list = []
            for instr in instrs.split('\n'):
                instrs_list.append(instr)
                acc_len += len(instr.split(' '))
            
            # convert the cleaned ingredients into a Python list
            ingrs_list = ast.literal_eval(ingrs)

            # discard recipes with too few or too many ingredients or instruction words
            if len(ingrs_list) < args.minnumingrs or len(ingrs_list) >= args.maxnumingrs \
                or len(instrs_list) < args.minnuminstrs or len(instrs_list) >= args.maxnuminstrs \
                or acc_len < args.minnumwords:
                continue

            # tokenize sentences + title and update counter
            for sentence in instrs_list:
                tokens = nltk.tokenize.word_tokenize(sentence)
                counter_toks.update(tokens)
            title = nltk.tokenize.word_tokenize(row['Title'].lower())
            counter_toks.update(title)
            counter_ingrs.update(ingrs_list)

        pickle.dump(counter_ingrs, open(args.save_path + 'allingrs_count.pkl', 'wb'))
        pickle.dump(counter_toks, open(args.save_path + 'allwords_count.pkl', 'wb'))

    # TODO: Make sure that this aligns with our current data
    # manually add missing entries for better clustering
    '''
    base_words = ['peppers', 'tomato', 'spinach_leaves', 'turkey_breast', 'lettuce_leaf',
                  'chicken_thighs', 'milk_powder', 'bread_crumbs', 'onion_flakes',
                  'red_pepper', 'pepper_flakes', 'juice_concentrate', 'cracker_crumbs', 'hot_chili',
                  'seasoning_mix', 'dill_weed', 'pepper_sauce', 'sprouts', 'cooking_spray', 'cheese_blend',
                  'basil_leaves', 'pineapple_chunks', 'marshmallow', 'chile_powder',
                  'cheese_blend', 'corn_kernels', 'tomato_sauce', 'chickens', 'cracker_crust',
                  'lemonade_concentrate', 'red_chili', 'mushroom_caps', 'mushroom_cap', 'breaded_chicken',
                  'frozen_pineapple', 'pineapple_chunks', 'seasoning_mix', 'seaweed', 'onion_flakes',
                  'bouillon_granules', 'lettuce_leaf', 'stuffing_mix', 'parsley_flakes', 'chicken_breast',
                  'basil_leaves', 'baguettes', 'green_tea', 'peanut_butter', 'green_onion', 'fresh_cilantro',
                  'breaded_chicken', 'hot_pepper', 'dried_lavender', 'white_chocolate',
                  'dill_weed', 'cake_mix', 'cheese_spread', 'turkey_breast', 'chucken_thighs', 'basil_leaves',
                  'mandarin_orange', 'laurel', 'cabbage_head', 'pistachio', 'cheese_dip',
                  'thyme_leave', 'boneless_pork', 'red_pepper', 'onion_dip', 'skinless_chicken', 'dark_chocolate',
                  'canned_corn', 'muffin', 'cracker_crust', 'bread_crumbs', 'frozen_broccoli',
                  'philadelphia', 'cracker_crust', 'chicken_breast']
    

    for base_word in base_words:
        if base_word not in counter_ingrs.keys():
            counter_ingrs[base_word] = 1
    '''

    # TODO: clean the dataset
    counter_ingrs, cluster_ingrs = cluster_ingredients(counter_ingrs)
    counter_ingrs, cluster_ingrs = remove_plurals(counter_ingrs, cluster_ingrs)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter_toks.items() if cnt >= args.threshold_words]
    ingrs = {word: cnt for word, cnt in counter_ingrs.items() if cnt >= args.threshold_ingrs}

    # Recipe vocab
    # Create a vocab wrapper and add some special tokens.
    vocab_toks = Vocabulary()
    vocab_toks.add_word('<start>')
    vocab_toks.add_word('<end>')
    vocab_toks.add_word('<eoi>') # end of recipe

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab_toks.add_word(word)
    vocab_toks.add_word('<pad>')

    # Ingredient vocab
    # Create a vocab wrapper for ingredients
    vocab_ingrs = Vocabulary()
    idx = vocab_ingrs.add_word('<end>')
    # this returns the next idx to add words to
    # Add the ingredients to the vocabulary.
    for k, _ in ingrs.items():
        for ingr in cluster_ingrs[k]:
            idx = vocab_ingrs.add_word(ingr, idx)
        idx += 1
    _ = vocab_ingrs.add_word('<pad>', idx)

    print("Total ingr vocabulary size: {}".format(len(vocab_ingrs)))
    print("Total token vocabulary size: {}".format(len(vocab_toks)))

    ## CHANGE:      
    dataset = {'train': [], 'val': [], 'test': []}

    ######
    # 2. Tokenize and build dataset based on vocabularies.
    ######
    IMAGE_DIR = '../archive/Food Images/'
    for i, row in dataset_df.iterrows():
        # get all instructions for this recipe
        instrs = row['Instructions']

        instrs_list = []
        acc_len = 0 # cumulative num of words
        instrs_list = []
        for instr in instrs.split('\n'):
            instrs_list.append(instr)
            acc_len += len(instr.split(' '))

        ingrs_list = ast.literal_eval(['Cleaned_Ingredients'])
        image_path = os.path.join(IMAGE_DIR, row['Image_Name'])
        id2im[i] = image_path

        # retrieve pre-detected ingredients for this entry
        labels = []

        # BUG: not sure what this block of code does
        '''
        for j, det_ingr in enumerate(det_ingrs):
            if len(det_ingr) > 0 and valid[j]:
                det_ingr_undrs = get_ingredient(det_ingr, replace_dict_ingrs)
                ingrs_list.append(det_ingr_undrs)
                label_idx = vocab_ingrs(det_ingr_undrs)
                if label_idx is not vocab_ingrs('<pad>') and label_idx not in labels:
                    labels.append(label_idx)
        '''

        # we discard recipes with too many or too few ingredients or instruction words
        if len(labels) < args.minnumingrs or len(labels) >= args.maxnumingrs \
                or len(instrs_list) < args.minnuminstrs or len(instrs_list) >= args.maxnuminstrs \
                or acc_len < args.minnumwords:
            continue

        # tokenize sentences
        toks = []

        for instr in instrs_list:
            tokens = nltk.tokenize.word_tokenize(instr)
            toks.append(tokens)

        title = nltk.tokenize.word_tokenize(row['Title'].lower())
        newentry = {'id': i, 'instructions': instrs_list, 'tokenized': toks,
                    'ingredients': ingrs_list, 'images': image_path, 'title': title}
        dataset[entry['partition']].append(newentry)

    print(f"Dataset size: {dataset.size}")

    return vocab_ingrs, vocab_toks, dataset


def main(args):

    vocab_ingrs, vocab_toks, dataset = build_vocab_epicurious(args)

    with open(os.path.join(args.save_path, args.suff+'epicurious_vocab_ingrs.pkl'), 'wb') as f:
        pickle.dump(vocab_ingrs, f)
    with open(os.path.join(args.save_path, args.suff+'epicurious_vocab_toks.pkl'), 'wb') as f:
        pickle.dump(vocab_toks, f)

    for split in dataset.keys():
        with open(os.path.join(args.save_path, args.suff+'epicurious_' + split + '.pkl'), 'wb') as f:
            pickle.dump(dataset[split], f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epicurious_path', type=str,
                        default='path/to/epicurious',
                        help='epicurious path')

    parser.add_argument('--save_path', type=str, default='../data/',
                        help='path for saving vocabulary wrapper')

    parser.add_argument('--suff', type=str, default='')

    parser.add_argument('--threshold_ingrs', type=int, default=10,
                        help='minimum ingr count threshold')

    parser.add_argument('--threshold_words', type=int, default=10,
                        help='minimum word count threshold')

    parser.add_argument('--maxnuminstrs', type=int, default=20,
                        help='max number of instructions (sentences)')

    parser.add_argument('--maxnumingrs', type=int, default=20,
                        help='max number of ingredients')

    parser.add_argument('--minnuminstrs', type=int, default=2,
                        help='max number of instructions (sentences)')

    parser.add_argument('--minnumingrs', type=int, default=2,
                        help='max number of ingredients')

    parser.add_argument('--minnumwords', type=int, default=20,
                        help='minimum number of characters in recipe')

    parser.add_argument('--forcegen', dest='forcegen', action='store_true')
    parser.set_defaults(forcegen=False)

    args = parser.parse_args()
    main(args)