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
from utils.metrics import softIoU, compute_metrics, update_error_types, MaskedCrossEntropyCriterion
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
        loss=lambda outputs, targets: MaskedCrossEntropyCriterion(outputs, targets, ignore_index=[instrs_vocab_size-1], reduce=False),
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

        for split in ['train', 'val']:

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
            total_loss = total_softIou = 0
            # print(f"Epoch {epoch}")
            # print()
            for i in range(total_step):

                img_inputs, captions, ingr_gt, img_ids, paths = next(loader)

                img_inputs = tf.convert_to_tensor(img_inputs.numpy())
                captions = tf.convert_to_tensor(captions.numpy())
                ingr_gt = tf.convert_to_tensor(ingr_gt.numpy())
                true_caps_batch = tf.constant(captions)[:, 1:]
                loss_dict = {}
                losses = {}

                if split == 'train':
                    targets = captions[:, 1:]
                    targets = tf.reshape(targets, [-1])

                    with tf.GradientTape() as tape:
                        # for train_var in model.trainable_variables:
                        #     print(train_var)
                        #     tape.watch(train_var)
                        # tape.watch(model.trainable_variables)
                        outputs = model(img_inputs, captions, ingr_gt, losses=losses, training=True)
                        losses = model.training_losses
                        loss = model.loss(outputs, targets)
                
                if split == 'val':
                    outputs = model(img_inputs, captions, ingr_gt, losses=losses)
                #     losses = model.validation_losses

                #     if not args.recipe_only:
                #         outputs = model(img_inputs, captions, ingr_gt, losses=losses, sample=True)

                #         ingr_ids_greedy = outputs['ingr_ids']

                #         mask = mask_from_eos(ingr_ids_greedy, eos_value=0, mult_before=False)
                #         ingr_ids_greedy = tf.where(mask == 0, ingr_vocab_size - 1, ingr_ids_greedy)
                #         pred_one_hot = label2onehot(ingr_ids_greedy, ingr_vocab_size - 1)
                #         target_one_hot = label2onehot(ingr_gt, ingr_vocab_size - 1)
                #         iou_sample = softIoU(pred_one_hot, target_one_hot)
                #         # iou_sample = tf.reduce_sum(iou_sample) / (tf.shape(iou_sample)[0] + 1e-6) 
                #         iou_sample = tf.reduce_sum(iou_sample) / (tf.math.count_nonzero(iou_sample) + 1e-6)
                #         loss_dict['iou_sample'] = iou_sample.numpy()

                #         update_error_types(error_types, pred_one_hot, target_one_hot)

                #         del outputs, pred_one_hot, target_one_hot, iou_sample
                
                # if not args.ingrs_only:
                #     recipe_loss = losses['recipe_loss']

                #     recipe_loss = tf.reshape(recipe_loss, tf.shape(true_caps_batch))
                #     non_pad_mask = tf.cast(tf.not_equal(true_caps_batch, instrs_vocab_size - 1), tf.float32)
                #     print("non_pad_mask shape", non_pad_mask.shape)

                #     print("recipe loss shape", recipe_loss.shape)
                #     recipe_loss_masked = tf.reduce_sum(recipe_loss*non_pad_mask, axis=-1) / tf.reduce_sum(non_pad_mask, axis=-1)
                #     perplexity = tf.exp(recipe_loss)

                #     recipe_loss = tf.reduce_mean(recipe_loss_masked)
                #     perplexity = tf.reduce_mean(perplexity)

                #     loss_dict['recipe_loss'] = recipe_loss.numpy()
                #     loss_dict['perplexity'] = perplexity.numpy()
                # else:
                #     recipe_loss = 0

                # if not args.recipe_only:

                #     ingr_loss = losses['ingr_loss']
                #     ingr_loss = tf.reduce_mean(ingr_loss)
                #     loss_dict['ingr_loss'] = ingr_loss.numpy()  

                #     eos_loss = losses['eos_loss']
                #     eos_loss = tf.reduce_mean(eos_loss)
                #     loss_dict['eos_loss'] = eos_loss.numpy() 

                #     iou_seq = losses['iou']
                #     iou_seq = tf.reduce_mean(iou_seq)
                #     loss_dict['iou'] = iou_seq.numpy() 

                #     card_penalty = losses['card_penalty']
                #     card_penalty = tf.reduce_mean(card_penalty)
                #     loss_dict['card_penalty'] = card_penalty.numpy() 
                # else:
                #     ingr_loss, eos_loss, card_penalty = 0, 0, 0

                # loss = args.loss_weight[0] * recipe_loss + args.loss_weight[1] * ingr_loss \
                #        + args.loss_weight[2]*eos_loss + args.loss_weight[3]*card_penalty
                # loss_dict['loss'] = loss.numpy()

                # for key in loss_dict.keys():
                #     total_loss_dict[key].append(loss_dict[key])

                if split == 'train':
                    gradients = tape.gradient(loss, model.trainable_variables)
                    for grad, var in zip(gradients, model.trainable_variables):
                        print(f"{var.name}, grad mean: {tf.reduce_mean(grad) if grad is not None else 'No gradient'}")
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    softIou = model.metrics[0](outputs, targets)

                    print("loss shape", loss.shape)
                    avg_batch_loss = tf.reduce_mean(loss)
                    avg_softIou = tf.reduce_mean(softIou)

                    total_loss += avg_batch_loss
                    total_softIou += avg_softIou
                    print(f"Batch: {i}; Loss: {avg_batch_loss}; SoftIou: {softIou}")

                if split == 'val':
                    print(outputs)

            print(f"Epoch: {epoch}; Average loss: {total_loss / total_step}; Average softIou: {total_softIou / total_step}")

                # # Print log info
                # if args.log_step != -1 and i % args.log_step == 0:
                #     elapsed_time = time.time()-start
                #     lossesstr = ""
                #     for k in total_loss_dict.keys():
                #         if len(total_loss_dict[k]) == 0:
                #             continue
                #         this_one = "%s: %.4f" % (k, np.mean(total_loss_dict[k][-args.log_step:]))
                #         lossesstr += this_one + ', '
                #     # this only displays nll loss on captions, the rest of losses will be in tensorboard logs
                #     strtoprint = 'Split: %s, Epoch [%d/%d], Step [%d/%d], Losses: %sTime: %.4f' % (split, epoch,
                #                                                                                    args.num_epochs, i,
                #                                                                                    total_step,
                #                                                                                    lossesstr,
                #                                                                                    elapsed_time)
                #     print(strtoprint)

                #     if args.tensorboard:
                #         # logger.histo_summary(model=model, step=total_step * epoch + i)
                #         logger.scalar_summary(mode=split+'_iter', epoch=total_step*epoch+i,
                #                               **{k: np.mean(v[-args.log_step:]) for k, v in total_loss_dict.items() if v})

                #     start = time.time()
                # del loss, losses, captions, img_inputs

            # if split == 'val' and not args.recipe_only:
            #     ret_metrics = {'accuracy': [], 'f1': [], 'jaccard': [], 'f1_ingredients': [], 'dice': []}
            #     compute_metrics(ret_metrics, error_types,
            #                     ['accuracy', 'f1', 'jaccard', 'f1_ingredients', 'dice'], eps=1e-10,
            #                     weights=None)

            #     total_loss_dict['f1'] = ret_metrics['f1']
            # if args.tensorboard:
            #     # 1. Log scalar values (scalar summary)
            #     logger.scalar_summary(mode=split,
            #                           epoch=epoch,
            #                           **{k: np.mean(v) for k, v in total_loss_dict.items() if v})

    if args.tensorboard:
        logger.close()


if __name__ == '__main__':
    args = get_parser()
    tf.random.set_seed(1234)
    np.random.seed(1234)
    main(args)
