import math
# import modules.utils as utils
from modules.multihead_attention import MultiheadAttention
import tensorflow as tf
import keras
# import tensorflow_addons as tfa
import numpy as np
import copy


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    # creates tensor from scratch - to avoid multigpu issues
    seq_length = tf.shape(tensor)[1]
    max_pos = padding_idx + 1 + seq_length
    #if not hasattr(make_positions, 'range_buf'):
    range_buf = tf.range(padding_idx + 1, max_pos, dtype=tensor.dtype)
    #make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    mask = tf.not_equal(tensor, padding_idx)
    positions = tf.broadcast_to(range_buf[:tensor.shape[1]], tensor.shape)
    if left_pad:
        positions = positions - mask.size(1) + tf.reduce_sum(tf.cast(mask, tf.int32), axis=1, keepdims=True)

    out = tf.where(mask, positions, tensor)
    return out


class LearnedPositionalEmbedding(tf.keras.Layer):
    """This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.embeddings = self.add_weight(name="embeddings", shape=(num_embeddings, embedding_dim), initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=embedding_dim**-0.5))

    def call(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = tf.fill([1, 1], self.padding_idx + tf.shape(input)[1])
        else:
            positions = make_positions(tf.cast(input, tf.int32), self.padding_idx, self.left_pad)
        return tf.nn.embedding_lookup(self.embeddings, positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1

class SinusoidalPositionalEmbedding(tf.keras.Layer):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = np.arange(num_embeddings)[:, np.newaxis] * emb[np.newaxis, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = np.concatenate([emb, np.zeros(num_embeddings, 1)], axis=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def call(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        # recompute/expand embeddings if needed
        bsz, seq_len = tf.shape(input)[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > tf.shape(self.weights)[0]:
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            return tf.tile(self.weights[self.padding_idx + seq_len, tf.newaxis, :], [tf.shape(input)[0], 1, 1])

        positions = make_positions(tf.cast(input, tf.int32), self.padding_idx, self.left_pad)
        return tf.gather(self.weights, positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
    

# Layer helpers

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim, padding_idx=padding_idx)
    weights = tf.random.normal(shape=(num_embeddings, embedding_dim), mean=0.0, stddev=embedding_dim ** -0.5)
    weights = tf.tensor_scatter_nd_update(weights, tf.constant([[padding_idx]]), tf.zeros((1, embedding_dim)))
    m.set_weights([weights])
    return m

def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        weights = tf.random.normal(shape=(num_embeddings, embedding_dim), mean=0.0, stddev=embedding_dim ** -0.5)
        weights = tf.tensor_scatter_nd_update(weights, tf.constant([[padding_idx]]), tf.zeros((1, embedding_dim)))
        m.set_weights([weights])
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings)
    return m

class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Decoder layer block."""

    def __init__(self, embed_dim, n_att, dropout=0.5, normalize_before=True, last_ln=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.relu_dropout = dropout
        self.normalize_before = normalize_before

        head_dim = self.embed_dim // n_att
        assert head_dim * n_att == self.embed_dim, "embed_dim must be divisible by num_heads"
        # self-attention on generated recipe
        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=n_att,
            key_dim=head_dim,
            value_dim=head_dim,
            dropout=dropout
        )

        # conditional-attention on img_features
        self.cond_attn = keras.layers.MultiHeadAttention(
            num_heads=n_att,
            key_dim=head_dim,
            value_dim=head_dim,
            dropout=dropout
        )

        self.fc1 = keras.layers.Dense(units=self.embed_dim,kernel_initializer='glorot_uniform', bias_initializer='zeros')
        self.fc2 = keras.layers.Dense(units=self.embed_dim,kernel_initializer='glorot_uniform', bias_initializer='zeros')
        self.layer_norms = [keras.layers.LayerNormalization(epsilon=1e-5) for i in range(3)]
        self.dropout_layer = keras.layers.Dropout(rate=self.dropout)

        self.use_last_ln = last_ln
        if self.use_last_ln:
            self.last_ln = keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, ingr_features, ingr_mask, incremental_state, img_features, training=False):
        # self attention
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        # print("x shape", x.shape)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            training=training
        )

        x = self.dropout_layer(x, training=training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        # conditional attention
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        if ingr_features is None:
            img_features = tf.transpose(img_features, perm= [1, 2, 0])
            # NOTE: img_features shape should be (batch_size, img_features, embed_size)
            # print("img inputs shape", img_features.shape)
            # img features here is populated with numbers
            x = self.cond_attn(query=x,
                    key=img_features,
                    value=img_features,
                    training=training
                )
            # some rows here have been replaced with nan 

        elif img_features is None:
            # NOTE: ingr_features shape should be (batch_size, seq_len, embed_size)
            # print("img inputs shape", img_features.shape)
            x = self.cond_attn(query=x,
                    key=ingr_features,
                    value=ingr_features,
                    training= training
                )
            
        else:
            # attention on concatenation of encoder_out and encoder_aux, query self attn (x)
            img_features = tf.transpose(img_features, perm=[1, 2, 0])
            ingr_features = tf.transpose(ingr_features, perm=[1,0,2])
            # NOTE: img_features shape should be (batch_size, img_features, embed_size)
            # print("img inputs shape", img_features.shape)
            kv = tf.concat((img_features, ingr_features), axis=1)
            # mask = tf.concat([tf.zeros((img_features.shape[1], img_features.shape[0]), dtype=tf.int32), tf.cast(ingr_mask, tf.int32)], axis=1)
            x = self.cond_attn(query=x,
                    key=kv,
                    value=kv,
                    training=training
                )

        x = self.dropout_layer(x, training=training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        # Fully connected layers
        residual = x
        x = self.maybe_layer_norm(-1, x, before=True)
        x = self.fc1(x)
        x = self.dropout_layer(x, training=training)
        x = self.fc2(x)
        x = self.dropout_layer(x, training=training)
        x = residual + x
        x = self.maybe_layer_norm(-1, x, after=True)

        if self.use_last_ln:
            x = self.last_ln(x)

        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class DecoderTransformer(tf.keras.Model):
    """Transformer decoder."""

    def __init__(self, embed_size, vocab_size, dropout=0.5, seq_length=20, num_instrs=15,
                 attention_nheads=16, pos_embeddings=True, num_layers=16, learned=True, normalize_before=True,
                 normalize_inputs=False, last_ln=False, scale_embed_grad=False):
        super(DecoderTransformer, self).__init__()
        # print(num_layers, "num_layers")
        self.dropout = dropout
        self.dropout_layer = keras.layers.Dropout(self.dropout)
        self.seq_length = seq_length * num_instrs
        self.embed_tokens = keras.layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embed_size, 
            embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=embed_size**-0.5),
            name="decoder_transformer_embed_tokens")
        self.final_ln = keras.layers.LayerNormalization(epsilon=1e-5, name='final_layer_norm_decoder_transformer')
        if pos_embeddings:
            self.embed_positions = PositionalEmbedding(1024, embed_size, padding_idx=0, left_pad=False, learned=learned)
        else:
            self.embed_positions = None
        self.normalize_inputs = normalize_inputs
        if self.normalize_inputs:
            self.layer_norms_in = [keras.layers.LayerNormalization(epsilon=1e-5) for i in range(3)]

        self.embed_scale = math.sqrt(embed_size)
        self.td_layers = [TransformerDecoderLayer(embed_size, attention_nheads, dropout, normalize_before, last_ln) for _ in range(num_layers)]
        self.linear = keras.layers.Dense(units=vocab_size-1,kernel_initializer='glorot_uniform', bias_initializer='zeros')

    def call(self, ingr_features, ingr_mask, captions, img_features, incremental_state=None, training = False):
        if ingr_features is not None:
            ingr_features = tf.transpose(ingr_features, perm=[0, 2, 1])
            ingr_features = tf.transpose(ingr_features, perm=[1, 0, 2])
            if self.normalize_inputs:
                self.layer_norms_in[0](ingr_features)

        if img_features is not None:
            img_features = tf.transpose(img_features, perm=[0, 2, 1])
            img_features = tf.transpose(img_features, perm=[1, 0, 2])
            if self.normalize_inputs:
                self.layer_norms_in[1](img_features)

        if ingr_mask is not None:
            ingr_mask = tf.squeeze(ingr_mask, axis=1)
            ingr_mask = 1 - ingr_mask
            ingr_mask = tf.cast(ingr_mask, tf.uint8)

        # embed positions
        if self.embed_positions is not None:
            positions = self.embed_positions(captions, incremental_state=incremental_state)
        if incremental_state is not None:
            if self.embed_positions is not None:
                positions = positions[:, -1:]
            captions = captions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(captions)
        if self.embed_positions is not None:
            x += positions

        if self.normalize_inputs:
            x = self.layer_norms_in[2](x)

        x = self.dropout_layer(x, training=training)

        # B x T x C -> T x B x C
        x = tf.transpose(x, perm=[1, 0] + list(range(2, tf.rank(x))))
        x = tf.transpose(x, perm=[1, 0, 2])
        for p, layer in enumerate(self.td_layers):
            # print("Transformer Layer", p)
            x = layer(
                x,
                ingr_features=ingr_features,
                ingr_mask=ingr_mask,
                incremental_state=incremental_state,
                img_features=img_features,
                training = training
            )
        
        # T x B x C -> B x T x C
        x = tf.transpose(x, perm=[1, 0] + list(range(2, tf.rank(x))))
        x = tf.transpose(x, perm=[1, 0, 2])
        x = self.linear(x)
        predicted = tf.argmax(x, axis=-1)

        return x, predicted

    def sample(self, ingr_features, ingr_mask, greedy=True, temperature=1.0, beam=-1,
               img_features=None, first_token_value=0,
               replacement=True, last_token_value=0, training = False):
        
        incremental_state = {}

        # create dummy previous word
        fs = tf.shape(ingr_features)[0] if ingr_features is not None else tf.shape(img_features)[0]

        if beam != -1:
            if fs == 1:
                # print('beam search')
                return self.sample_beam(ingr_features, ingr_mask, beam, img_features, first_token_value,
                                        replacement, last_token_value, training = training)
            else:
                print ("Beam Search can only be used with batch size of 1. Running greedy or temperature sampling...")

        first_word = np.ones(fs)*first_token_value

        first_word = tf.fill([fs], first_token_value)
        first_word = tf.cast(first_word, tf.int64)
        sampled_ids = [first_word]
        logits = []

        for i in range(self.seq_length):
            # forward
            caption_ids = [tf.cast(id, tf.int32) for id in sampled_ids]
            captions = tf.stack(caption_ids, axis=1)

            # print("incremental state", incremental_state)
            # print("captions", captions)
            # print("caption ids", caption_ids)
            outputs, _ = self.call(ingr_features=ingr_features, ingr_mask=ingr_mask, captions=captions, img_features=img_features, incremental_state=incremental_state, training=training)
            # print("outputs shape", outputs.shape)
            outputs = tf.squeeze(outputs, axis=1)
            if not replacement:
                # predicted mask
                if i == 0:
                    predicted_mask = tf.zeros_like(outputs)
                else:
                    # ensure no repetitions in sampling if replacement==False
                    batch_ind = [j for j in range(fs) if sampled_ids[i][j] != 0]
                    if len(batch_ind) != 0:
                        sampled_ids_new = tf.gather(sampled_ids[i], batch_ind)
                        predicted_mask = np.zeros_like(outputs)
                        for b_ind, sample_id in zip(batch_ind, sampled_ids_new):
                            predicted_mask[b_ind, sample_id] = -np.inf
                        predicted_mask = tf.convert_to_tensor(predicted_mask)
                    else:
                        predicted_mask = tf.zeros_like(outputs)

                # mask previously selected ids
                outputs += predicted_mask

            logits.append(outputs)
            if greedy:
                outputs_prob = tf.nn.softmax(outputs, axis=-1)
                predicted = tf.argmax(outputs_prob, axis=1)
            else:
                k = 10
                # outputs_prob = outputs / temperature
                outputs_prob = tf.squeeze(outputs, axis=1) / temperature
                outputs_prob = tf.nn.softmax(outputs_prob, axis=-1)

                # top k random sampling
                prob_prev_topk, indices = tf.nn.top_k(outputs_prob, k=k)
                # predicted = tf.random.categorical(tf.math.log(prob_prev_topk), 1)
                predicted = tf.reshape(tf.random.categorical(tf.math.log(prob_prev_topk), 1), -1)
                # predicted = tf.gather_nd(indices, predicted, batch_dims=1)
                predicted = tf.gather_nd(indices, predicted, batch_dims=1)[:, 0]

            sampled_ids.append(predicted)
            # print(sampled_ids.shape)
            # print(predicted.shape)
            # tf.concat([sampled_ids, predicted], axis=0)

        sampled_ids = tf.stack(sampled_ids[1:], axis=1)
        logits = tf.stack(logits, axis=1)

        return sampled_ids, logits

    def sample_beam(self, ingr_features, ingr_mask, beam=3, img_features=None, first_token_value=0,
                   replacement=True, last_token_value=0, training = False):
        k = beam
        alpha = 0.0
        # create dummy previous word
        fs = tf.shape(ingr_features)[0] if ingr_features is not None else tf.shape(img_features)[0]
        first_word = tf.fill([fs], first_token_value)
        first_word = tf.cast(first_word, tf.int64)

        sequences = [[[first_word], 0, {}, False, 1]]
        finished = []

        for i in range(self.seq_length):
            # forward
            all_candidates = []
            for rem in range(len(sequences)):
                incremental = sequences[rem][2]
                outputs, _ = self.call(ingr_features, ingr_mask, tf.stack(sequences[rem][0], axis=1), img_features, incremental, training=training)
                outputs = tf.squeeze(outputs, axis=1)
                if not replacement:
                    # predicted mask
                    if i == 0:
                        predicted_mask = tf.zeros_like(outputs)
                    else:
                        # ensure no repetitions in sampling if replacement==False
                        batch_ind = [j for j in range(fs) if sequences[rem][0][i][j] != 0]
                        sampled_ids_new = sequences[rem][0][i][batch_ind]
                        predicted_mask[batch_ind, sampled_ids_new] = float('-inf')

                    # mask previously selected ids
                    outputs += predicted_mask

                outputs_prob = tf.nn.log_softmax(outputs, axis=-1)
                probs, indices = tf.math.top_k(outputs_prob, k=beam)
                # tokens is [batch x beam ] and every element is a list
                # score is [ batch x beam ] and every element is a scalar
                # incremental is [batch x beam ] and every element is a dict

                for bid in range(beam):
                    tokens = sequences[rem][0] + [indices[:, bid]]
                    # score = sequences[rem][1] + probs[:, bid]
                    score = sequences[rem][1] + tf.get_static_value(tf.squeeze(probs[:, bid]))
                    if tf.get_static_value(indices[:,bid]) == last_token_value:
                        finished.append([tokens, score, None, True, sequences[rem][-1] + 1])
                    else:
                        all_candidates.append([tokens, score, incremental, False, sequences[rem][-1] + 1])

            # if all the top-k scoring beams have finished, we can return them
            ordered_all = sorted(all_candidates + finished, key=lambda tup: tup[1]/(np.power(tup[-1],alpha)),
                                 reverse=True)[:k]
            if all(el[-1] == True for el in ordered_all):
                all_candidates = []

            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1]/(np.power(tup[-1],alpha)), reverse=True)
            # select k best
            sequences = ordered[:k]
            finished = sorted(finished,  key=lambda tup: tup[1]/(np.power(tup[-1],alpha)), reverse=True)[:k]

        if len(finished) != 0:
            sampled_ids = tf.stack(finished[0][0][1:], axis=1)
            logits = finished[0][1]
        else:
            sampled_ids = tf.stack(sequences[0][0][1:], axis=1)
            logits = sequences[0][1]
        return sampled_ids, logits

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = tf.zeros([])
        return state_dict