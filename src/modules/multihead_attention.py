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

        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.in_proj_weight)
    #     nn.init.xavier_uniform_(self.out_proj.weight)
    #     if self.in_proj_bias is not None:
    #         nn.init.constant_(self.in_proj_bias, 0.)
    #         nn.init.constant_(self.out_proj.bias, 0.)

    def call(self, query, key, value, key_padding_mask=None, training=False, 
                mask_future_timesteps=False):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        batch_size, seq_length, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        qkv = tf.linalg.matmul(query, self.in_proj_weight, transpose_b=True) + self.in_proj_bias
        q, k, v = tf.split(qkv, 3, axis=-1)
        q *= self.scaling

        q = tf.reshape(q, (batch_size, seq_length, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, seq_length, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, seq_length, self.num_heads, self.head_dim))
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 3, 1])
        v = tf.transpose(v, [0, 2, 1, 3])

        # attn_weights = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        attn_weights = tf.matmul(q, k)

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps:
            mask_shape = [1, 1, seq_length, seq_length]
            future_mask = tf.linalg.band_part(tf.ones(mask_shape), -1, 0)
            attn_weights -= 1e9 * (1 - future_mask)
        if key_padding_mask is not None:
            mask = tf.reshape(mask, [batch_size, 1, 1, seq_length])
            attn_weights += (mask * -1e9)
        # CHANGED: F.softmax --> tf.nn
        # normalize attention scores
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = tf.nn.dropout(attn_weights, rate=self.dropout if training else 0.0)

        #CHANGED: torch.bmm --> tf.matmul
        # apply attention to get weight average
        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, seq_length, self.embed_dim])

        output = self.out_proj(attn_output)

        return output, attn_weights
