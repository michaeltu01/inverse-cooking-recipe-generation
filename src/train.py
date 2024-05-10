# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from args import get_parser
import tensorflow as tf
import numpy as np
import os
import random
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import get_model
import sys
import json
import time
from utils.tb_visualizer import Visualizer
# import tensorflow_addons as tfa
from model import mask_from_eos, label2onehot
from utils.metrics import softIoU, compute_metrics, update_error_types
import random

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main(args):

    # Create model directory & other aux folders for logging
    where_to_save = os.path.join(args.save_dir, args.project_name, args.model_name)
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    logs_dir = os.path.join(where_to_save, 'logs')
    tb_logs = os.path.join(args.save_dir, args.project_name, 'tb_logs', args.model_name)
    make_dir(where_to_save)
    make_dir(logs_dir)
    make_dir(checkpoints_dir)
    make_dir(tb_logs)

    if args.tensorboard:
        logger = Visualizer(tb_logs, name='visual_results')

    # Build data loader
    data_loaders = {}
    datasets = {}

    data_dir = args.epicurious_dir
    
    for split in ['train', 'val']: 

        if split == 'train':
            transform = tf.keras.Sequential([
                tf.keras.layers.Resizing(args.image_size, args.image_size),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomCrop(args.crop_size, args.crop_size),
                tf.keras.layers.Rescaling(1./255),
                tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])
            ])
        elif split == 'val':
            transform = tf.keras.Sequential([
                tf.keras.layers.Resizing(args.image_size, args.image_size),
                tf.keras.layers.CenterCrop(args.crop_size, args.crop_size),
                tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])
            ])

        max_num_samples = max(args.max_eval, args.batch_size) if split == 'val' else -1
        data_loaders[split], datasets[split] = get_loader(data_dir, args.aux_data_dir, split,
                                                        args.maxseqlen, args.maxnuminstrs,
                                                        args.maxnumlabels, args.maxnumims,
                                                        transform, args.batch_size, shuffle=(split == 'train'),
                                                        num_workers=args.num_workers,
                                                        drop_last=True, max_num_samples=max_num_samples,
                                                        use_lmdb=args.use_lmdb, suff=args.suff)

    print("split", split)
    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    instrs_vocab_size = datasets[split].get_instrs_vocab_size()

    # Build the model
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[softIoU]
    )

    print("InverseCookingModel training variables", model.trainable_variables)

    # add model parameters
    if args.ingrs_only:
        params = model.ingredient_decoder.trainable_variables
    elif args.recipe_only:
        params = model.recipe_decoder.trainable_variables + model.ingredient_encoder.trainable_variables
    else:
        params = model.recipe_decoder.trainable_variables + model.ingredient_decoder.trainable_variables \
                + model.ingredient_encoder.trainable_variables
    # only train the linear layer in the encoder if we are not transfering from another model
    # NOTE: Lowkey unnecessary
    if args.transfer_from == '':
        params += model.image_encoder.linear.trainable_variables
  

    print("CNN params:", sum(tf.size(p).numpy() for p in model.image_encoder.trainable_variables))
    print("Decoder params:", sum(tf.size(p).numpy() for p in params))

    # Train the model
    for epoch in range(args.num_epochs):

        total_step = len(data_loaders[split])
        loader = iter(data_loaders[split])

        start = time.time()
        total_loss = 0
        print(f"Epoch {epoch}")
        print()
        for i in range(total_step):

            img_inputs, captions, ingr_gt, img_ids, paths = next(loader)

            img_inputs = tf.convert_to_tensor(img_inputs.numpy())
            captions = tf.convert_to_tensor(captions.numpy())
            ingr_gt = tf.convert_to_tensor(ingr_gt.numpy())

            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                losses = model(img_inputs, captions, ingr_gt, training=True)            

            gradients = tape.gradient(losses['iou'], model.trainable_variables)
            for grad, var in zip(gradients, model.trainable_variables):
                print(f"{var.name}, grad mean: {tf.reduce_mean(grad) if grad is not None else 'No gradient'}")
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += losses['iou']
            print(f"Batch: {i}; Loss: {losses['iou']}")
        print(f"Epoch: {epoch}; Average loss: {total_loss / total_step}")

    if args.tensorboard:
        logger.close()


if __name__ == '__main__':
    args = get_parser()
    tf.random.set_seed(1234)
    np.random.seed(1234)
    main(args)
