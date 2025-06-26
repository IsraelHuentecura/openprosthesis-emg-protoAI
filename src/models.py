import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# --- Custom Attention Layer ---
class AdditiveAttention(layers.Layer):
    """
    Additive Attention mechanism, as proposed for the CRNN-Attn model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape: tuple):
        hidden_units = input_shape[-1]
        self.W1 = self.add_weight(name="W1", shape=(hidden_units, hidden_units), initializer="glorot_uniform")
        self.b1 = self.add_weight(name="b1", shape=(hidden_units,), initializer="zeros")
        self.W2 = self.add_weight(name="W2", shape=(hidden_units, 1), initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Calculate attention scores
        score = tf.tanh(tf.tensordot(x, self.W1, axes=1) + self.b1)
        score = tf.tensordot(score, self.W2, axes=1)
        
        # Calculate attention weights
        attention_weights = tf.nn.softmax(tf.squeeze(score, axis=-1), axis=1)
        
        # Apply weights to the input sequence
        context_vector = tf.reduce_sum(x * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return context_vector

# === MODEL DEFINITIONS =======================================================

def build_dualstream_original(
    raw_shape: tuple, feat_dim: int, t_subwin: int, n_cls: int,
    lstm_units: int = 200, conv_filters: int = 256,
    dropout_rate: float = 0.3, learning_rate: float = 1e-4
) -> models.Model:
    """
    Builds the original DualStream-Original model, faithful to the paper.
    Features a raw data stream and a handcrafted features stream, fused into Bi-LSTMs.
    """
    in_raw = layers.Input(shape=(t_subwin, *raw_shape), name="raw")
    in_feat = layers.Input(shape=(t_subwin, feat_dim), name="feat")

    # Raw Data Stream
    r = layers.TimeDistributed(layers.Reshape((raw_shape[0], raw_shape[1])))(in_raw)
    r = layers.TimeDistributed(layers.Conv1D(conv_filters, 3, padding="same", activation="relu"))(r)
    r = layers.TimeDistributed(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))(r)
    r = layers.TimeDistributed(layers.Conv1D(conv_filters, 3, padding="same", activation="relu"))(r)
    r = layers.TimeDistributed(layers.GlobalAveragePooling1D())(r)
    
    # Feature Stream
    f = layers.Reshape((t_subwin, feat_dim, 1))(in_feat)
    f = layers.TimeDistributed(layers.Conv1D(conv_filters, 3, padding="same", activation="relu"))(f)
    f = layers.TimeDistributed(layers.Conv1D(conv_filters, 3, padding="same", activation="relu"))(f)
    f = layers.TimeDistributed(layers.GlobalAveragePooling1D())(f)

    # Fusion and Temporal Block
    x = layers.Concatenate()([r, f])
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=dropout_rate))(x)

    # Classifier Head
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(n_cls, activation="softmax", name="output")(x)
    
    model = models.Model(inputs=[in_raw, in_feat], outputs=out, name="DualStream-Original")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_dualstream_lite(
    raw_shape: tuple, feat_dim: int, t_subwin: int, n_cls: int,
    raw_branch_filters: int = 64, feat_branch_filters: int = 64,
    lstm_units: int = 128, dropout_rate: float = 0.3, learning_rate: float = 1e-3
) -> models.Model:
    """
    Builds the DualStream-Lite model, a lightweight adaptation of the DualStream concept.
    """
    in_raw = layers.Input((t_subwin, *raw_shape), name="raw")
    in_feat = layers.Input((t_subwin, feat_dim), name="feat")

    r = layers.TimeDistributed(layers.Reshape((raw_shape[0], raw_shape[1])))(in_raw)
    r = layers.TimeDistributed(layers.Conv1D(raw_branch_filters, 3, padding="same", activation="relu"))(r)
    r = layers.TimeDistributed(layers.GlobalAveragePooling1D())(r)

    f = layers.TimeDistributed(layers.Dense(feat_branch_filters, activation="relu"))(in_feat)

    x = layers.Concatenate()([r, f])
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=dropout_rate))(x)

    out = layers.Dense(n_cls, activation="softmax")(x)

    model = models.Model([in_raw, in_feat], out, name="DualStream-Lite")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_emghandnet_original(
    raw_shape: tuple, t_subwin: int, n_cls: int,
    filters: tuple = (64, 64, 64, 64), lstm_units: int = 200,
    dropout_rate: float = 0.3, learning_rate: float = 1e-3
) -> models.Model:
    """
    Builds the original EMGHandNet model, faithful to the paper.
    Uses 1D convolutions and stacked Bi-LSTMs.
    """
    inp = layers.Input(shape=(t_subwin, *raw_shape), name="raw")
    x = layers.TimeDistributed(layers.Reshape((raw_shape[0], raw_shape[1])))(inp)

    for i, f in enumerate(filters):
        x = layers.TimeDistributed(layers.Conv1D(f, 3, padding="same", activation="relu"), name=f'td_conv1d_{i}')(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name=f'td_bn_{i}')(x)
        x = layers.TimeDistributed(layers.MaxPool1D(2), name=f'td_pool_{i}')(x)
        
    x = layers.TimeDistributed(layers.Flatten(), name='td_flatten')(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=dropout_rate))(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(n_cls, activation="softmax", name="output")(x)
    
    model = models.Model(inputs=inp, outputs=out, name="EMGHandNet-Original")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_emghandnet_2d(
    raw_shape: tuple, t_subwin: int, n_cls: int,
    filters: tuple = (64, 128), lstm_units: int = 128,
    dropout_rate: float = 0.3, learning_rate: float = 1e-3
) -> models.Model:
    """
    Builds the EMGHandNet-2D model, which treats sEMG windows as images using Conv2D layers.
    """
    inp = layers.Input((t_subwin, *raw_shape), name="raw")
    x = inp
    for f in filters:
        x = layers.TimeDistributed(layers.Conv2D(f, (3, 3), padding="same", activation="relu"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPool2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=dropout_rate))(x)
    out = layers.Dense(n_cls, activation="softmax")(x)

    model = models.Model(inp, out, name="EMGHandNet-2D")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_hyt_net(
    raw_shape: tuple, feat_dim: int, t_subwin: int, n_cls: int,
    num_heads: int = 4, projection_dim: int = 128,
    dropout_rate: float = 0.3, learning_rate: float = 1e-3
) -> models.Model:
    """
    Builds the Hybrid Transformer Network (HyT-Net) architecture.
    It combines a CNN branch for raw data with a Transformer Encoder for sequence modeling.
    """
    # CNN Branch for spatial feature extraction from each window
    seg_in = layers.Input(raw_shape, name="segment_input")
    x_cnn = layers.Conv2D(64, (5, 1), padding="same", activation="relu")(seg_in)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.Conv2D(64, (3, 1), padding="same", activation="relu")(x_cnn)
    x_cnn = layers.BatchNormalization()(x_cnn)
    seg_vec = layers.GlobalAveragePooling2D()(x_cnn)
    segment_cnn = models.Model(seg_in, seg_vec, name="segment_cnn")

    # Main Model Inputs
    in_raw = layers.Input((t_subwin, *raw_shape), name="raw")
    in_feat = layers.Input((t_subwin, feat_dim), name="feat")
    
    # Process sequences
    raw_seq = layers.TimeDistributed(segment_cnn)(in_raw)
    feat_seq = layers.TimeDistributed(layers.Dense(64, activation="relu"))(in_feat)
    
    # Fusion and Projection
    fusion = layers.Concatenate(axis=-1)([raw_seq, feat_seq])
    projection = layers.Dense(projection_dim, activation="linear")(fusion)

    # Transformer Encoder Block
    def transformer_encoder_block(seq_input):
        x = layers.LayerNormalization(epsilon=1e-6)(seq_input)
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)(x, x)
        res = x + seq_input
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Dense(projection_dim * 4, activation="relu")(x)
        x = layers.Dense(projection_dim)(x)
        return x + res

    # Stacked Transformer Blocks
    z = transformer_encoder_block(projection)
    z = transformer_encoder_block(z)

    # Classifier Head
    z = layers.GlobalAveragePooling1D()(z)
    z = layers.Dropout(dropout_rate)(z)
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(dropout_rate)(z)
    out = layers.Dense(n_cls, activation="softmax", name="output")(z)
    
    model = models.Model(inputs=[in_raw, in_feat], outputs=out, name="HyT-Net")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_crnn_attn(
    raw_shape: tuple, feat_dim: int, t_subwin: int, n_cls: int,
    alpha: float = 0.75, gru_units: int = 128,
    dropout_rate: float = 0.3, learning_rate: float = 2e-3, label_smoothing: float = 0.1
) -> models.Model:
    """
    Builds the CRNN-Attn model, featuring separable convolutions for efficiency,
    a Bi-GRU layer, and an Additive Attention mechanism.
    """
    # Lightweight CNN branch using separable convolutions
    inp_cnn = layers.Input(raw_shape)
    x_cnn = layers.SeparableConv2D(int(32 * alpha), 3, padding="same", activation="relu")(inp_cnn)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.SeparableConv2D(int(64 * alpha), 3, padding="same", activation="relu")(x_cnn)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.GlobalAveragePooling2D()(x_cnn)
    cnn_branch = models.Model(inp_cnn, x_cnn, name="mobile_cnn")

    # Main Model
    in_raw = layers.Input((t_subwin, *raw_shape), name="raw")
    in_feat = layers.Input((t_subwin, feat_dim), name="feat")
    
    raw_seq = layers.TimeDistributed(cnn_branch)(in_raw)
    feat_seq = layers.TimeDistributed(layers.Dense(96, activation="relu"))(in_feat)
    
    merged_seq = layers.Concatenate()([raw_seq, feat_seq])
    
    x = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True, dropout=dropout_rate))(merged_seq)
    x = AdditiveAttention()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    out = layers.Dense(n_cls, activation="softmax", kernel_regularizer=regularizers.l2(1e-4))(x)
    
    model = models.Model([in_raw, in_feat], out, name="CRNN-Attn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
                  metrics=["accuracy"])
    return model