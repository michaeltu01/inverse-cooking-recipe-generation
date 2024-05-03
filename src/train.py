# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from args import get_parser
import tensorflow as tf
import torch.nn as nn
import torch.autograd as autograd
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
import torch.backends.cudnn as cudnn
from utils.tb_visualizer import Visualizer
from model import mask_from_eos, label2onehot
from utils.metrics import softIoU, compute_metrics, update_error_types
import random

device = 'GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'

def merge_models(args, model, ingr_vocab_size, instrs_vocab_size):
    load_args = pickle.load(open(os.path.join(args.save_dir, args.project_name,
                                              args.transfer_from, 'checkpoints/args.pkl'), 'rb'))
    
    model_ingrs = get_model(load_args, ingr_vocab_size, instrs_vocab_size)
    model_path = os.path.join(args.save_dir, args.project_name, args.transfer_from, 'checkpoints', 'modelbest.ckpt')
    
    # Load the trained model parameters
    model_ingrs.load_weights(model_path)  # TensorFlow model loading
    model.ingredient_decoder = model_ingrs.ingredient_decoder
    args.transf_layers_ingrs = load_args.transf_layers_ingrs
    args.n_att_ingrs = load_args.n_att_ingrs

    return args, model

def save_model(model, optimizer, checkpoints_dir, suff=''):
    model_save_path = os.path.join(checkpoints_dir, 'model' + suff + '.ckpt')
    optimizer_save_path = os.path.join(checkpoints_dir, 'optim' + suff + '.ckpt')

    # Save the model weights
    model.save_weights(model_save_path)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.save(optimizer_save_path)


def count_parameters(model):
    return np.sum([tf.size(variable).numpy() for variable in model.trainable_variables])

def set_lr(optimizer, decay_factor):
    if hasattr(optimizer, 'lr'):
        optimizer.lr = optimizer.lr * decay_factor
    elif hasattr(optimizer, '_decayed_lr'):
        lr = optimizer._decayed_lr().numpy()
        optimizer.lr = lr * decay_factor

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_and_preprocess_image(image_path, is_train, image_size, crop_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])

    if is_train:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        image = tf.image.pad_to_bounding_box(image, 10, 10, crop_size + 20, crop_size + 20)
        image = tf.image.crop_to_bounding_box(image, 10, 10, crop_size, crop_size)
    else:
        image = tf.image.resize_with_crop_or_pad(image, crop_size, crop_size)

    # Normalize image
    image = tf.cast(image, tf.float32) / 255.0
    image -= tf.constant([0.485, 0.456, 0.406])
    image /= tf.constant([0.229, 0.224, 0.225])
    
    return image

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

    # check if we want to resume from last checkpoint of current model
    if args.resume:
        args = pickle.load(open(os.path.join(checkpoints_dir, 'args.pkl'), 'rb'))
        args.resume = True

    # logs to disk
    if not args.log_term:
        print("Training logs will be saved to:", os.path.join(logs_dir, 'train.log'))
        sys.stdout = open(os.path.join(logs_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(logs_dir, 'train.err'), 'w')

    print(args)
    pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))

    # patience init
    curr_pat = 0

    # Build data loader
    data_loaders = {}
    datasets = {}

    data_dir = args.epicurious_dir

    for split in ['train', 'val']:
        max_num_samples = max(args.max_eval, args.batch_size) if split == 'val' else -1
        data_loaders[split], datasets[split] = get_loader(data_dir, args.aux_data_dir, split,
                                                        args.maxseqlen, args.maxnuminstrs,
                                                        args.maxnumlabels, args.maxnumims,
                                                        None, args.batch_size, shuffle=(split == 'train'),
                                                        num_workers=args.num_workers,
                                                        drop_last=True, max_num_samples=max_num_samples,
                                                        use_lmdb=args.use_lmdb, suff=args.suff)

    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    instrs_vocab_size = datasets[split].get_instrs_vocab_size()

    # Build the model
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
    decay_factor = 1.0

    # add model parameters
    if args.ingrs_only:
        params = model.ingredient_decoder.trainable_variables
    elif args.recipe_only:
        params = model.recipe_decoder.trainable_variables + model.ingredient_encoder.trainable_variables
    else:
        params = model.recipe_decoder.trainable_variables + model.ingredient_decoder.trainable_variables \
                + model.ingredient_encoder.trainable_variables

    # only train the linear layer in the encoder if we are not transfering from another model
    if args.transfer_from == '':
        params += model.image_encoder.linear.trainable_variables
    params_cnn = model.image_encoder.resnet.trainable_variables
  

    print("CNN params:", sum(tf.size(p).numpy() for p in model.image_encoder.trainable_variables))
    print("Decoder params:", sum(tf.size(p).numpy() for p in params))

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    if args.resume:
        model_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints', 'modelbest.ckpt')
        optim_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints', 'optimbest.ckpt')
        model.load_weights(model_path)
        optimizer.load_weights(optim_path)

    if args.transfer_from != '':
        model_path = os.path.join(args.save_dir, args.project_name, args.transfer_from, 'checkpoints', 'modelbest.ckpt')
        pretrained_model = get_model(args, ingr_vocab_size, instrs_vocab_size)
        pretrained_model.load_weights(model_path)
        model.image_encoder.set_weights(pretrained_model.image_encoder.get_weights())

        args, model = merge_models(args, model, ingr_vocab_size, instrs_vocab_size)

    if device != 'cpu' and len(tf.config.experimental.list_physical_devices('GPU')) > 1:
        model = tf.keras.utils.multi_gpu_model(model, gpus=tf.config.experimental.list_physical_devices('GPU'))


    if not hasattr(args, 'current_epoch'):
        args.current_epoch = 0

    es_best = 10000 if args.es_metric == 'loss' else 0
    # Train the model
    start = args.current_epoch


    for epoch in range(start, args.num_epochs):

        # save current epoch for resuming
        if args.tensorboard:
            logger.reset()

        args.current_epoch = epoch
        # increase / decrase values for moving params
        if args.decay_lr:
            frac = epoch // args.lr_decay_every
            decay_factor = args.lr_decay_rate ** frac
            new_lr = args.learning_rate*decay_factor
            print ('Epoch %d. lr: %.5f'%(epoch, new_lr))
            optimizer.learning_rate.assign(new_lr)

        if args.finetune_after != -1 and args.finetune_after < epoch \
                and not keep_cnn_gradients and params_cnn is not None:

            print("Starting to fine tune CNN")
            # start with learning rates as they were (if decayed during training)
            # optimizer = torch.optim.Adam([{'params': params},
            #                               {'params': params_cnn,
            #                                'lr': decay_factor*args.learning_rate*args.scale_learning_rate_cnn}],
            #                              lr=decay_factor*args.learning_rate)
            optimizer.learning_rate.assign(decay_factor * args.learning_rate)
            keep_cnn_gradients = True

        for split in ['train', 'val']:

            if split == 'train':
                model.train()
            else:
                model.eval()
            total_step = len(data_loaders[split])
            loader = iter(data_loaders[split])

            total_loss_dict = {'recipe_loss': [], 'ingr_loss': [],
                               'eos_loss': [], 'loss': [],
                               'iou': [], 'perplexity': [], 'iou_sample': [],
                               'f1': [],
                               'card_penalty': []}

            error_types = {'tp_i': 0, 'fp_i': 0, 'fn_i': 0, 'tn_i': 0,
                           'tp_all': 0, 'fp_all': 0, 'fn_all': 0}

            start = time.time()

            for i in range(total_step):

                img_inputs, captions, ingr_gt, img_ids, paths = loader.next()

                img_inputs = tf.convert_to_tensor(img_inputs.numpy())
                captions = tf.convert_to_tensor(captions.numpy())
                ingr_gt = tf.convert_to_tensor(ingr_gt.numpy())
                true_caps_batch = captions[:, 1:]

                loss_dict = {}

                if split == 'val':
                    predictions, outputs = model(img_inputs, captions, ingr_gt, training=False)

                    if not args.recipe_only:
                        ingr_ids_greedy = outputs['ingr_ids']

                        mask = mask_from_eos(ingr_ids_greedy, eos_value=0, mult_before=False)
                        ingr_ids_greedy = tf.where(mask == 0, ingr_vocab_size - 1, ingr_ids_greedy)
                        pred_one_hot = label2onehot(ingr_ids_greedy, ingr_vocab_size - 1)
                        target_one_hot = label2onehot(ingr_gt, ingr_vocab_size - 1)

                        iou_sample = softIoU(pred_one_hot, target_one_hot)
                        iou_sample = tf.reduce_sum(iou_sample) / (tf.size(iou_sample) + 1e-6) 
                        loss_dict['iou_sample'] = iou_sample.numpy()

                        update_error_types(error_types, pred_one_hot, target_one_hot)

                        del outputs, pred_one_hot, target_one_hot, iou_sample

                else:
                    with tf.GradientTape() as tape:
                        loss = model(img_inputs, captions, ingr_gt, training=True)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                if not args.ingrs_only:
                    recipe_loss = loss['recipe_loss']

                    recipe_loss = tf.reshape(recipe_loss, tf.shape(true_caps_batch))
                    non_pad_mask = tf.cast(tf.not_equal(true_caps_batch, instrs_vocab_size - 1), tf.float32)

                    recipe_loss_masked = tf.reduce_sum(recipe_loss * non_pad_mask, axis=-1) / tf.reduce_sum(non_pad_mask, axis=-1)

                    perplexity = tf.exp(recipe_loss_masked)

                    recipe_loss = tf.reduce_mean(recipe_loss_masked)
                    perplexity = tf.reduce_mean(perplexity)

                    loss_dict['recipe_loss'] = recipe_loss
                    loss_dict['perplexity'] = perplexity
                else:
                    recipe_loss = 0

                if not args.recipe_only:
                    ingr_loss = losses['ingr_loss']
                    ingr_loss = tf.reduce_mean(ingr_loss)
                    loss_dict['ingr_loss'] = ingr_loss.numpy()  

                    eos_loss = losses['eos_loss']
                    eos_loss = tf.reduce_mean(eos_loss)
                    loss_dict['eos_loss'] = eos_loss.numpy() 

                    iou_seq = losses['iou']
                    iou_seq = tf.reduce_mean(iou_seq)
                    loss_dict['iou'] = iou_seq.numpy() 

                    card_penalty = losses['card_penalty']
                    card_penalty = tf.reduce_mean(card_penalty)
                    loss_dict['card_penalty'] = card_penalty.numpy() 
                else:
                    ingr_loss, eos_loss, card_penalty = 0, 0, 0

                # Combining weighted losses to compute the total loss
                total_loss = (args.loss_weight[0] * recipe_loss +
                            args.loss_weight[1] * ingr_loss +
                            args.loss_weight[2] * eos_loss +
                            args.loss_weight[3] * card_penalty)

                loss_dict['loss'] = total_loss.numpy()

                for key in loss_dict.keys():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = []
                    total_loss_dict[key].append(loss_dict[key].numpy())

                if split == 'train':
                    with tf.GradientTape() as tape:
                        total_loss = sum(loss_dict.values())
                    gradients = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                if args.log_step != -1 and i % args.log_step == 0:
                    elapsed_time = time.time() - start
                    lossesstr = ""
                    for k in total_loss_dict.keys():
                        if len(total_loss_dict[k]) == 0:
                            continue
                        this_one = "%s: %.4f" % (k, np.mean(total_loss_dict[k][-args.log_step:]))
                        lossesstr += this_one + ', '
                    strtoprint = 'Split: %s, Epoch [%d/%d], Step [%d/%d], Losses: %sTime: %.4f' % (
                        split, epoch, args.num_epochs, i, total_step, lossesstr, elapsed_time)
                    print(strtoprint)

                    if args.tensorboard:
                        with logger.as_default():
                            for k, v in total_loss_dict.items():
                                if len(v) > 0:
                                    tf.summary.scalar(k, np.mean(v[-args.log_step:]), step=total_step * epoch + i)
                            tf.summary.flush()

                    start = time.time()
                del loss, losses, captions, img_inputs

            if split == 'val' and not args.recipe_only:
                ret_metrics = {'accuracy': [], 'f1': [], 'jaccard': [], 'f1_ingredients': [], 'dice': []}
                compute_metrics(ret_metrics, error_types,
                                ['accuracy', 'f1', 'jaccard', 'f1_ingredients', 'dice'], eps=1e-10,
                                weights=None)

                total_loss_dict['f1'] = ret_metrics['f1']
            if args.tensorboard:
                # 1. Log scalar values (scalar summary)
                logger.scalar_summary(mode=split,
                                      epoch=epoch,
                                      **{k: np.mean(v) for k, v in total_loss_dict.items() if v})

        # Save the model's best checkpoint if performance was improved
        es_value = np.mean(total_loss_dict[args.es_metric])

        # save current model as well
        save_model(model, optimizer, checkpoints_dir, suff='')
        if (args.es_metric == 'loss' and es_value < es_best) or (args.es_metric == 'iou_sample' and es_value > es_best):
            es_best = es_value
            save_model(model, optimizer, checkpoints_dir, suff='best')
            pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))
            curr_pat = 0
            print('Saved checkpoint.')
        else:
            curr_pat += 1

        if curr_pat > args.patience:
            break

    if args.tensorboard:
        logger.close()


if __name__ == '__main__':
    args = get_parser()
    tf.random.set_seed(1234)
    np.random.seed(1234)
    main(args)
