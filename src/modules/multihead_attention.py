import tensorflow as tf

class MultiheadAttention(tf.keras.layers.Layer):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self._mask = None

        #CHANGED these shapes are directly reflected in initializing the weight below

        # self.in_proj_weight = tf.Variable(tf.Tensor(3*embed_dim, embed_dim))
        # if bias:
        #     self.in_proj_bias = tf.Variable(tf.Tensor(3*embed_dim))
        # else:
        #     self.register_parameter('in_proj_bias', None)

        #CHANGED: replaced functionality of reset_parameters() - which was just to initialize the weights
        self.in_proj_weight = self.add_weight(
            shape=(3 * embed_dim, embed_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='in_proj_weight'
        )
        if bias:
            self.in_proj_bias = self.add_weight(
                shape=(3 * embed_dim,),
                initializer='zeros',
                trainable=True,
                name='in_proj_bias'
            )
        else:
            self.in_proj_bias = None

        # # CHANGED: nn.Linear --> Dense layer with bias - not sure about this one?
        # TODO: may have to specifiy that input size is embed_size
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias)

        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.in_proj_weight)
    #     nn.init.xavier_uniform_(self.out_proj.weight)
    #     if self.in_proj_bias is not None:
    #         nn.init.constant_(self.in_proj_bias, 0.)
    #         nn.init.constant_(self.out_proj.bias, 0.)

    def call(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, training=False):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # tf.debugging.check_numerics(query, "query nan found")
        qkv_same = id(query) == id(key) == id(value)
        kv_same = id(key) == id(value)
        
        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        
        if qkv_same:
            q, k, v = self.in_proj_qkv(query)

        elif kv_same:
            
            q = self.in_proj_q(query)
 
            # print(k, "k")
            # print(v, "v")
            if key is None:
                assert value is None
                k = v = tf.constant(0, dtype=q.dtype)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        src_len = tf.shape(k)[0]

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = tf.transpose(q, [1, 0, 2])
        k = tf.transpose(k, [1, 0, 2])
        v = tf.transpose(v, [1, 0, 2])

        q = tf.reshape(q, [bsz * self.num_heads, -1, self.head_dim])
        k = tf.reshape(k, [bsz * self.num_heads, -1, self.head_dim])
        v = tf.reshape(v, [bsz * self.num_heads, -1, self.head_dim])

        attn_weights = tf.linalg.matmul(q, k, transpose_b=True)
        # print(attn_weights, "mha attn weights output")

        if mask_future_timesteps and incremental_state is None:
            # assert tf.shape(query) == tf.shape(key), \
            assert all([tf.shape(query)[i] == tf.shape(key)[i] for i in range(len(tf.shape(query)))]), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += tf.expand_dims(self.buffered_mask(attn_weights), axis=0)

        if key_padding_mask is not None:
            key_padding_mask = tf.cast(key_padding_mask, tf.float32)
            mask = key_padding_mask[:, tf.newaxis, tf.newaxis, :]
            adder = (1.0 - mask) * -1e9
            k += adder
            v += adder

        attn_weights = tf.nn.softmax(attn_weights, axis = -1)
        attn_weights = self.dropout_layer(attn_weights, training=training)

        attn = tf.linalg.matmul(attn_weights, v)
        attn = tf.reshape(attn, [bsz, -1, self.embed_dim])
        attn = tf.transpose(attn, [1, 0, 2])
        attn = self.out_proj(attn)

        attn_weights = tf.reshape(attn_weights, [bsz, self.num_heads, tf.shape(query)[0], -1])
        attn_weights = tf.reduce_mean(attn_weights, axis=1)

        # print(attn, "mha attn output")
        

        return attn, attn_weights

        # # --------------------
        # qkv = tf.linalg.matmul(query, self.in_proj_weight, transpose_b=True) + self.in_proj_bias
        # q, k, v = tf.split(qkv, 3, axis=-1)
        # q *= self.scaling

        # q = tf.reshape(q, (batch_size, seq_length, self.num_heads, self.head_dim))
        # k = tf.reshape(k, (batch_size, seq_length, self.num_heads, self.head_dim))
        # v = tf.reshape(v, (batch_size, seq_length, self.num_heads, self.head_dim))
        # q = tf.transpose(q, [0, 2, 1, 3])
        # k = tf.transpose(k, [0, 2, 3, 1])
        # v = tf.transpose(v, [0, 2, 1, 3])

        # # attn_weights = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        # attn_weights = tf.matmul(q, k)

        # # only apply masking at training time (when incremental state is None)
        # if mask_future_timesteps:
        #     mask_shape = [1, 1, seq_length, seq_length]
        #     future_mask = tf.linalg.band_part(tf.ones(mask_shape), -1, 0)
        #     attn_weights -= 1e9 * (1 - future_mask)
        # if key_padding_mask is not None:
        #     mask = tf.reshape(mask, [batch_size, 1, 1, seq_length])
        #     attn_weights += (mask * -1e9)
        # # CHANGED: F.softmax --> tf.nn
        # # normalize attention scores
        # attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        # attn_weights = self.dropout_layer(self.dropout, training=)

        # #CHANGED: torch.bmm --> tf.matmul
        # # apply attention to get weight average
        # attn_output = tf.matmul(attn_weights, v)
        # attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        # attn_output = tf.reshape(attn_output, [batch_size, seq_length, self.embed_dim])

        # output = self.out_proj(attn_output)

        # return output, attn_weights
    
    def in_proj_qkv(self, query):
        return tf.split(self._in_proj(query), num_or_size_splits=3, axis=-1)

    def in_proj_kv(self, key):
        # return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)
        return tf.split(self._in_proj(key, start=self.embed_dim), num_or_size_splits=2, axis=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2*self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2*self.embed_dim)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        # return F.linear(input, weight, bias)
        # input = tf.transpose(input, perm= [1, 2, 0])
        result = tf.linalg.matmul(input, weight, transpose_b=True)
        if bias is not None:
            result += bias
        return result

    def fill_with_neg_inf(self, t):
        """FP16-compatible function that fills a tensor with -inf."""
        output = tf.fill(tf.shape(t), tf.constant(float('-inf')))
        return output

    def buffered_mask(self, tensor):
        dim = tf.shape(tensor)[-1]
        if self._mask is None:
            self._mask = tf.linalg.band_part(self.fill_with_neg_inf(tf.zeros_like(tensor)), 0, -1)
        if tf.shape(self._mask)[0] < dim:
            # self._mask = tf.linalg.band_part(self.fill_with_neg_inf(tf.reshape(self._mask, [dim, dim])), 0, -1)
            new_mask = tf.zeros((dim, dim), dtype=tensor.dtype)
            self._mask = tf.linalg.band_part(self.fill_with_neg_inf(new_mask), 0, -1) - tf.linalg.band_part(new_mask, 0, 0)
        # return self._mask[:dim, :dim]
        return self._mask

    # def reorder_incremental_state(self, incremental_state, new_order):
    #     """Reorder buffered internal state (for incremental generation)."""
    #     input_buffer = self._get_input_buffer(incremental_state)
    #     if input_buffer is not None:
    #         for k in input_buffer.keys():
    #             input_buffer[k] = input_buffer[k].index_select(1, new_order)
    #         self._set_input_buffer(incremental_state, input_buffer)

    # def _get_input_buffer(self, incremental_state):
    #     return get_incremental_state(
    #         self,
    #         incremental_state,
    #         'attn_state',
    #     ) or {}

    # def _set_input_buffer(self, incremental_state, buffer):
    #     set_incremental_state(
    #         self,
    #         incremental_state,
    #         'attn_state',
    #         buffer,
