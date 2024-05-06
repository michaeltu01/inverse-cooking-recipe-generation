import tensorflow as tf
from modules.encoder import EncoderCNN, EncoderLabels
from modules.transformer_decoder import DecoderTransformer
from modules.multihead_attention import MultiheadAttention
from utils.metrics import softIoU, MaskedCrossEntropyCriterion
import numpy as np

# NOTE: Replaced torch check for cuda
device = tf.device('/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0')

def label2onehot(labels, pad_value):
    # input labels to one hot vector
    inp_ = tf.expand_dims(labels, axis=-1)
    one_hot = tf.one_hot(inp_, depth=pad_value + 1, axis=2)
    one_hot = tf.reduce_max(one_hot, axis=1)
    # remove pad position
    one_hot = one_hot[:, :-1]
    # eos position is always 0
    one_hot = tf.concat([tf.zeros_like(one_hot[:, :1]), one_hot[:, 1:]], axis=1)
    # one hot shape: (batch_size, vocab_size, 1)
    one_hot = tf.squeeze(one_hot, axis=-1)
    # one hot shape: (batch_size, vocab_size)
    return one_hot


def mask_from_eos(ids, eos_value, mult_before=True):
    # mask = torch.ones(ids,)).to(device).byte()
    # mask_aux = torch.ones(ids.size(0)).to(device).byte()

    mask = np.ones(ids.shape)
    mask_aux = np.ones(ids.shape[0])

    # find eos in ingredient prediction
    for idx in range(tf.shape(ids)[1]):
        # force mask to have 1s in the first position to avoid division by 0 when predictions start with eos
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (ids[:, idx] != eos_value).numpy()
        else:
            mask_aux = mask_aux * (ids[:, idx] != eos_value).numpy()
            mask[:, idx] = mask[:, idx] * mask_aux
    return tf.convert_to_tensor(mask)

def get_model(args, ingr_vocab_size, instrs_vocab_size):

    # build ingredients embedding
    encoder_ingrs = EncoderLabels(args.embed_size, ingr_vocab_size,
                                  args.dropout_encoder, scale_grad=False)
    # build image model
    encoder_image = EncoderCNN(args.embed_size, args.dropout_encoder, args.image_model)

    decoder = DecoderTransformer(args.embed_size, instrs_vocab_size,
                                 dropout=args.dropout_decoder_r, seq_length=args.maxseqlen,
                                 num_instrs=args.maxnuminstrs,
                                 attention_nheads=args.n_att, num_layers=args.transf_layers,
                                 normalize_before=True,
                                 normalize_inputs=False,
                                 last_ln=False,
                                 scale_embed_grad=False)

    ingr_decoder = DecoderTransformer(args.embed_size, ingr_vocab_size, dropout=args.dropout_decoder_i,
                                      seq_length=args.maxnumlabels,
                                      num_instrs=1, attention_nheads=args.n_att_ingrs,
                                      pos_embeddings=False,
                                      num_layers=args.transf_layers_ingrs,
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)
    
    decoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, weight_decay=args.weight_decay),
        loss='binary_crossentropy',
        metrics=[softIoU]
    )

    ingr_decoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, weight_decay=args.weight_decay),
        loss='binary_crossentropy',
        metrics=[softIoU]
    )

    # recipe loss
    criterion = MaskedCrossEntropyCriterion(ignore_index=[instrs_vocab_size-1], reduce=False)

    # ingredients loss
    # NOTE: Replaced torch.nn.BCELoss -> tf.keras.losses.BinaryCrossentropy
    label_loss = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')
    eos_loss = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')

    model = InverseCookingModel(encoder_ingrs, decoder, ingr_decoder, encoder_image,
                                crit=criterion, crit_ingr=label_loss, crit_eos=eos_loss,
                                pad_value=ingr_vocab_size-1,
                                ingrs_only=args.ingrs_only, recipe_only=args.recipe_only,
                                label_smoothing=args.label_smoothing_ingr)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[softIoU]
    )

    return model

class InverseCookingModel(tf.keras.Model):
    def __init__(self, ingredient_encoder, recipe_decoder, ingr_decoder, image_encoder,
                 crit=None, crit_ingr=None, crit_eos=None,
                 pad_value=0, ingrs_only=True,
                 recipe_only=False, label_smoothing=0.0):
        super(InverseCookingModel, self).__init__()
        self.ingredient_encoder = ingredient_encoder
        self.recipe_decoder = recipe_decoder
        self.image_encoder = image_encoder
        self.ingredient_decoder = ingr_decoder

        # NOTE: I have NO CLUE what these hyperparameters do LMAO
        self.crit = crit
        self.crit_ingr = crit_ingr
        self.pad_value = pad_value
        self.ingrs_only = ingrs_only
        self.recipe_only = recipe_only
        self.crit_eos = crit_eos
        self.label_smoothing = label_smoothing

    # Changed to a more familiar 'call' function, instead of 'forward'
    def call(self, img_inputs, captions, target_ingrs, sample=False, keep_cnn_gradients=False, training = False):
        if sample:
            return self.sample(img_inputs, greedy=True)

        targets = captions[:, 1:]
        targets = tf.reshape(targets, [-1])

        # print("img_inputs in image encoder", img_inputs)
        img_features = self.image_encoder(img_inputs, keep_cnn_gradients=keep_cnn_gradients)
        # print("image features shape out of encoder", img_features.shape)

        losses = {}
        target_one_hot = label2onehot(target_ingrs, self.pad_value)
        target_one_hot_smooth = label2onehot(target_ingrs, self.pad_value)
        # print("pad value", self.pad_value)
        # print("target ingrs", target_ingrs)
        # print("target one_hot", target_one_hot)
        # print("smooth (?) target one_hot", target_one_hot_smooth)

        # ingredient prediction
        if not self.recipe_only:

            # mask = tf.cast(target_one_hot_smooth == 1, dtype=tf.float32)
            # target_one_hot_smooth = mask * (1 - self.label_smoothing) + (1 - mask) * target_one_hot_smooth

            target_one_hot_smooth = tf.where(target_one_hot_smooth == 0,
                                 tf.cast(self.label_smoothing / tf.cast(tf.shape(target_one_hot_smooth)[-1], dtype=tf.float32), dtype=tf.float32),
                                 1-self.label_smoothing)
            # target_one_hot_smooth[target_one_hot_smooth == 1] = (1-self.label_smoothing)
            # target_one_hot_smooth[target_one_hot_smooth == 0] = self.label_smoothing / tf.shape(target_one_hot_smooth)[-1]

            # decode ingredients with transformer
            # autoregressive mode for ingredient decoder
            ingr_ids, ingr_logits = self.ingredient_decoder.sample(None, None, greedy=True,
                                                                   temperature=1.0, img_features=img_features,
                                                                   first_token_value=0, replacement=False)

            # NOTE: torch.nn.functional.softmax -> tf.nn.softmax
            ingr_logits = tf.nn.softmax(ingr_logits, axis=-1)

            # find idxs for eos ingredient
            # eos probability is the one assigned to the first position of the softmax
            eos = ingr_logits[:, :, 0]
            target_eos = ((target_ingrs == 0) ^ (target_ingrs == self.pad_value))

            eos_pos = (target_ingrs == 0)
            eos_head = ((target_ingrs != self.pad_value) & (target_ingrs != 0))

            # select transformer steps to pool from
            mask_perminv = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)
            ingr_probs = ingr_logits * tf.expand_dims(tf.cast(mask_perminv, tf.float32), axis=-1)

            # NOTE: Replaced torch.max with tf.reduce_max
            ingr_probs = tf.reduce_max(ingr_probs, axis=1)

            # ignore predicted ingredients after eos in ground truth
            ingr_ids = tf.where(mask_perminv == 0,
                                self.pad_value,
                                ingr_ids)

            ingr_loss = self.crit_ingr(ingr_probs, target_one_hot_smooth)

            # NOTE: Replaced torch.mean with tf.reduce_mean
            ingr_loss = tf.math.reduce_mean(ingr_loss)

            losses['ingr_loss'] = ingr_loss

            # cardinality penalty
            # NOTE: Replaced torch.abs -> tf.math.abs
            losses['card_penalty'] = tf.math.abs(tf.reduce_sum(ingr_probs*target_one_hot, axis=1)) - tf.reduce_sum(target_one_hot, axis=1) + \
                                     tf.math.abs(tf.reduce_sum(ingr_probs*(1-target_one_hot), axis=1))

            eos_loss = self.crit_eos(eos, tf.cast(target_eos, tf.float32))

            mult = 1/2
            # eos loss is only computed for timesteps <= t_eos and equally penalizes 0s and 1s
            # losses['eos_loss'] = mult*(eos_loss * eos_pos.float()).sum(1) / (eos_pos.float().sum(1) + 1e-6) + \
            #                      mult*(eos_loss * eos_head.float()).sum(1) / (eos_head.float().sum(1) + 1e-6)
            #casting boolean masks to floats
            eos_pos_float = tf.cast(eos_pos, tf.float32)
            eos_head_float = tf.cast(eos_head, tf.float32)

            eos_pos_sum = tf.reduce_sum(eos_loss * eos_pos_float, axis=1)
            eos_pos_count = tf.reduce_sum(eos_pos_float, axis=1) + 1e-6

            eos_head_sum = tf.reduce_sum(eos_loss * eos_head_float, axis=1)
            eos_head_count = tf.reduce_sum(eos_head_float, axis=1) + 1e-6

            eos_loss_pos = mult * (eos_pos_sum / eos_pos_count)
            eos_loss_head = mult * (eos_head_sum / eos_head_count)
            
            # Sum up the loss terms
            losses['eos_loss'] = eos_loss_pos + eos_loss_head
            #END OF MS changes
            # iou
            pred_one_hot = label2onehot(ingr_ids, self.pad_value)
            # iou sample during training is computed using the true eos position
            losses['iou'] = softIoU(pred_one_hot, target_one_hot)

        if self.ingrs_only:
            return losses

        # encode ingredients
        target_ingr_feats = self.ingredient_encoder(target_ingrs)
        target_ingr_mask = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)

        target_ingr_mask = tf.expand_dims(tf.cast(target_ingr_mask, tf.float32), axis=1)

        outputs, ids = self.recipe_decoder(target_ingr_feats, target_ingr_mask, captions, img_features, training=training)

        outputs = outputs[:, :-1, :]
        outputs = tf.reshape(outputs, [tf.shape(outputs)
        [0] * tf.shape(outputs)[1], -1])

        loss = self.crit(outputs, targets) # MaskedCrossEntropyCriterion takes outputs, then targets

        losses['recipe_loss'] = loss

        return losses
    
    def sample(self, img_inputs, greedy=True, temperature=1.0, beam=-1, true_ingrs=None):

        outputs = dict()

        img_features = self.image_encoder(img_inputs)

        if not self.recipe_only:
            ingr_ids, ingr_probs = self.ingredient_decoder.sample(None, None, greedy=True, temperature=temperature,
                                                                  beam=-1,
                                                                  img_features=img_features, first_token_value=0,
                                                                  replacement=False)

            # mask ingredients after finding eos
            sample_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
            ingr_ids[sample_mask == 0] = self.pad_value

            outputs['ingr_ids'] = ingr_ids
            outputs['ingr_probs'] = ingr_probs.data

            mask = sample_mask
            input_mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=1)
            input_feats = self.ingredient_encoder(ingr_ids)

        if self.ingrs_only:
            return outputs

        # option during sampling to use the real ingredients and not the predicted ones to infer the recipe
        if true_ingrs is not None:
            input_mask = mask_from_eos(true_ingrs, eos_value=0, mult_before=False)
            true_ingrs[input_mask == 0] = self.pad_value
            input_feats = self.ingredient_encoder(true_ingrs)
            input_mask = tf.expand_dims(input_mask, axis=1)

        ids, probs = self.recipe_decoder.sample(input_feats, input_mask, greedy, temperature, beam, img_features, 0,
                                                last_token_value=1)

        outputs['recipe_probs'] = probs.data
        outputs['recipe_ids'] = ids

        return outputs
