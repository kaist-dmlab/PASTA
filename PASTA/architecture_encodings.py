import math

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, Input, Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Conv2D, Conv3D, MaxPool2D, MaxPool3D, Concatenate, LeakyReLU, Flatten, ReLU, Reshape, MultiHeadAttention, Dropout, LayerNormalization, GlobalAveragePooling1D, Softmax, LSTM, GRU, SimpleRNN
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        return config

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        config = super(Sampling, self).get_config()
        return config

def PASTA_CAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])
    
    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_network_ops) # channel ==> encoder/decoder
    H_n = MaxPool2D(3, padding = 'same')(H_n)
    H_n = Conv2D(1, 1, activation = 'relu')(H_n)
    H_n = Flatten()(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_temp_ops) # channel ==> encoder/decoder
    H_t = MaxPool2D(3, padding = 'same')(H_t)
    H_t = Conv2D(1, 1, activation = 'relu')(H_t)
    H_t = Flatten()(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)
    
    H_ec = Conv3D(16, 3, activation = 'relu', data_format = 'channels_first')(in_enc_connection) # channel ==> n_ae
    H_ec = MaxPool3D(5, padding = 'same')(H_ec)
    H_ec = Conv3D(1, 1, activation = 'relu')(H_ec)
    H_ec = Flatten()(H_ec)
    H_ec = Dense(16, activation = 'relu')(H_ec)
    H_ec = LeakyReLU()(H_ec)
    
    H_dc = Conv3D(16, 3, activation = 'relu', data_format = 'channels_first')(in_dec_connection) # channel ==> n_ae
    H_dc = MaxPool3D(5, padding = 'same')(H_dc)
    H_dc = Conv3D(1, 1, activation = 'relu')(H_dc)
    H_dc = Flatten()(H_dc)
    H_dc = Dense(16, activation = 'relu')(H_dc)
    H_dc = LeakyReLU()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")            
    
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")
    outputs = decoder(H)
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_CAE")
    return model, encoder, decoder
    
    
def PASTA_CVAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])
    
    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_network_ops) # channel ==> encoder/decoder
    H_n = MaxPool2D(3, padding = 'same')(H_n)
    H_n = Conv2D(1, 1, activation = 'relu')(H_n)
    H_n = Flatten()(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_temp_ops) # channel ==> encoder/decoder
    H_t = MaxPool2D(3, padding = 'same')(H_t)
    H_t = Conv2D(1, 1, activation = 'relu')(H_t)
    H_t = Flatten()(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)
    
    H_ec = Conv3D(16, 3, activation = 'relu', data_format = 'channels_first')(in_enc_connection) # channel ==> n_ae
    H_ec = MaxPool3D(5, padding = 'same')(H_ec)
    H_ec = Conv3D(1, 1, activation = 'relu')(H_ec)
    H_ec = Flatten()(H_ec)
    H_ec = Dense(16, activation = 'relu')(H_ec)
    H_ec = LeakyReLU()(H_ec)
    
    H_dc = Conv3D(16, 3, activation = 'relu', data_format = 'channels_first')(in_dec_connection) # channel ==> n_ae
    H_dc = MaxPool3D(5, padding = 'same')(H_dc)
    H_dc = Conv3D(1, 1, activation = 'relu')(H_dc)
    H_dc = Flatten()(H_dc)
    H_dc = Dense(16, activation = 'relu')(H_dc)
    H_dc = LeakyReLU()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        
        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)

        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)

        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        
        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))
        
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
            
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")
            
    # Define VAE model.
    outputs = decoder(z)
    vae = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_CVAE")

    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)
    
    return vae, encoder, decoder

    
def PASTA_TAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    
    seq_length, n_heads, ff_dim, dropout = connection_shape[0][1], 8, z_dim, 0.1
    
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])
    
    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Reshape([2 * 5, 10])(in_network_ops)
    H_n = TransformerBlock(embed_dim=10, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_n) # channel ==> encoder/decoder
    H_n = GlobalAveragePooling1D(data_format="channels_first")(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Reshape([2 * 6, 4])(in_temp_ops)
    H_t = TransformerBlock(embed_dim=4, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_t) # channel ==> encoder/decoder
    H_t = GlobalAveragePooling1D(data_format="channels_first")(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)
    
    H_ec = [] # hidden outputs from Transform for each AE (total n_ae = 5)
    for n in range(5):
        H_ec_n = Reshape([seq_length, 20])(in_enc_connection[:,n,:])
        H_ec_n = TransformerBlock(embed_dim=20, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_ec_n)
        H_ec_n = GlobalAveragePooling1D(data_format="channels_first")(H_ec_n)
        H_ec.append(H_ec_n)
    H_ec = Concatenate()(H_ec)
    
    H_dc = [] # hidden outputs from Transform for each AE
    for n in range(5):
        H_dc_n = Reshape([seq_length, 20])(in_dec_connection[:,n,:])
        H_dc_n = TransformerBlock(embed_dim=20, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_dc_n)
        H_dc_n = GlobalAveragePooling1D(data_format="channels_first")(H_dc_n)
        H_dc.append(H_dc_n)
    H_dc = Concatenate()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")     
        
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")        
    outputs = decoder(H)
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_TAE")
    return model, encoder, decoder


def PASTA_TVAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    
    seq_length, n_heads, ff_dim, dropout = connection_shape[0][1], 8, z_dim, 0.1
    
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])
    
    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Reshape([2 * 5, 10])(in_network_ops)
    H_n = TransformerBlock(embed_dim=10, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_n) # channel ==> encoder/decoder
    H_n = GlobalAveragePooling1D(data_format="channels_first")(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Reshape([2 * 6, 4])(in_temp_ops)
    H_t = TransformerBlock(embed_dim=4, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_t) # channel ==> encoder/decoder
    H_t = GlobalAveragePooling1D(data_format="channels_first")(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)
    
    H_ec = [] # hidden outputs from Transform for each AE (total n_ae = 5)
    for n in range(5):
        H_ec_n = Reshape([seq_length, 20])(in_enc_connection[:,n,:])
        H_ec_n = TransformerBlock(embed_dim=20, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_ec_n)
        H_ec_n = GlobalAveragePooling1D(data_format="channels_first")(H_ec_n)
        H_ec.append(H_ec_n)
    H_ec = Concatenate()(H_ec)
    
    H_dc = [] # hidden outputs from Transform for each AE
    for n in range(5):
        H_dc_n = Reshape([seq_length, 20])(in_dec_connection[:,n,:])
        H_dc_n = TransformerBlock(embed_dim=20, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_dc_n)
        H_dc_n = GlobalAveragePooling1D(data_format="channels_first")(H_dc_n)
        H_dc.append(H_dc_n)
    H_dc = Concatenate()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        
        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)

        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)

        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        
        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))
        
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")           
    
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")    
    
    # Define VAE model.
    outputs = decoder(z)
    vae = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_TVAE")

    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)
    
    return vae, encoder, decoder

    
def PASTA_CTAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    
    seq_length, n_heads, ff_dim, dropout = connection_shape[0][1], 8, z_dim, 0.1
    
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])

    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_network_ops) # channel ==> encoder/decoder
    H_n = MaxPool2D(3, padding = 'same')(H_n)
    H_n = Conv2D(1, 1, activation = 'relu')(H_n)
    H_n = Flatten()(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_temp_ops) # channel ==> encoder/decoder
    H_t = MaxPool2D(3, padding = 'same')(H_t)
    H_t = Conv2D(1, 1, activation = 'relu')(H_t)
    H_t = Flatten()(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)    

    H_ec = [] # hidden outputs from Transform for each AE (total n_ae = 5)
    for n in range(5):
        H_ec_n = Reshape([seq_length, 20])(in_enc_connection[:,n,:])
        H_ec_n = TransformerBlock(embed_dim=20, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_ec_n)
        H_ec_n = GlobalAveragePooling1D(data_format="channels_first")(H_ec_n)
        H_ec.append(H_ec_n)
    H_ec = Concatenate()(H_ec)
    
    H_dc = [] # hidden outputs from Transform for each AE
    for n in range(5):
        H_dc_n = Reshape([seq_length, 20])(in_dec_connection[:,n,:])
        H_dc_n = TransformerBlock(embed_dim=20, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_dc_n)
        H_dc_n = GlobalAveragePooling1D(data_format="channels_first")(H_dc_n)
        H_dc.append(H_dc_n)
    H_dc = Concatenate()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")              
    
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")     
    outputs = decoder(H)
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_CTAE")
    return model, encoder, decoder    
    
    
def PASTA_CTVAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    
    seq_length, n_heads, ff_dim, dropout = connection_shape[0][1], 8, z_dim, 0.1
        
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])

    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_network_ops) # channel ==> encoder/decoder
    H_n = MaxPool2D(3, padding = 'same')(H_n)
    H_n = Conv2D(1, 1, activation = 'relu')(H_n)
    H_n = Flatten()(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_temp_ops) # channel ==> encoder/decoder
    H_t = MaxPool2D(3, padding = 'same')(H_t)
    H_t = Conv2D(1, 1, activation = 'relu')(H_t)
    H_t = Flatten()(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)
    
    H_ec = [] # hidden outputs from Transform for each AE (total n_ae = 5)
    for n in range(5):
        H_ec_n = Reshape([seq_length, 20])(in_enc_connection[:,n,:])
        H_ec_n = TransformerBlock(embed_dim=20, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_ec_n)
        H_ec_n = GlobalAveragePooling1D(data_format="channels_first")(H_ec_n)
        H_ec.append(H_ec_n)
    H_ec = Concatenate()(H_ec)
    
    H_dc = [] # hidden outputs from Transform for each AE
    for n in range(5):
        H_dc_n = Reshape([seq_length, 20])(in_dec_connection[:,n,:])
        H_dc_n = TransformerBlock(embed_dim=20, num_heads=n_heads, ff_dim=ff_dim, rate=dropout)(H_dc_n)
        H_dc_n = GlobalAveragePooling1D(data_format="channels_first")(H_dc_n)
        H_dc.append(H_dc_n)
    H_dc = Concatenate()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        
        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)

        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)

        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        
        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))
        
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")            
    
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")
        
    # Define VAE model.
    outputs = decoder(z)
    vae = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_CTVAE")

    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)
    
    return vae, encoder, decoder

    
    
def PASTA_DAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])
    
    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Flatten()(in_network_ops) # channel ==> encoder/decoder
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Flatten()(in_temp_ops) # channel ==> encoder/decoder
    H_t = Dense(16, activation = 'relu')(H_t)
    
    H_ec = Flatten()(in_enc_connection) # channel ==> n_ae
    H_ec = Dense(16, activation = 'relu')(H_ec)
    H_ec = LeakyReLU()(H_ec)
    
    H_dc = Flatten()(in_dec_connection) # channel ==> n_ae
    H_dc = Dense(16, activation = 'relu')(H_dc)
    H_dc = LeakyReLU()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")      

    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")     
    outputs = decoder(H)
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_DAE")
    return model, encoder, decoder    
    
    
def PASTA_DVAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])
    
    H_g = Dense(16, activation = 'relu')(in_setting)

    H_n = Flatten()(in_network_ops) # channel ==> encoder/decoder
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Flatten()(in_temp_ops) # channel ==> encoder/decoder
    H_t = Dense(16, activation = 'relu')(H_t)
    
    H_ec = Flatten()(in_enc_connection) # channel ==> n_ae
    H_ec = Dense(16, activation = 'relu')(H_ec)
    H_ec = LeakyReLU()(H_ec)
    
    H_dc = Flatten()(in_dec_connection) # channel ==> n_ae
    H_dc = Dense(16, activation = 'relu')(H_dc)
    H_dc = LeakyReLU()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        
        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)

        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)

        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))        
        
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        
        z_mean = Dense(z_dim, name="z_mean")(H)
        z_log_var = Dense(z_dim, name="z_log_var")(H)
        z = Sampling()((z_mean, z_log_var))
        
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")    
    
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")
        
    # Define VAE model.
    outputs = decoder(z)
    vae = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_DVAE")

    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)
    
    return vae, encoder, decoder



################ AFTER REBUTTAL ##################
def PASTA_CRAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    
    seq_length, n_heads, ff_dim, dropout = connection_shape[0][1], 8, z_dim, 0.1
    
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])

    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_network_ops) # channel ==> encoder/decoder
    H_n = MaxPool2D(3, padding = 'same')(H_n)
    H_n = Conv2D(1, 1, activation = 'relu')(H_n)
    H_n = Flatten()(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_temp_ops) # channel ==> encoder/decoder
    H_t = MaxPool2D(3, padding = 'same')(H_t)
    H_t = Conv2D(1, 1, activation = 'relu')(H_t)
    H_t = Flatten()(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)    

    H_ec = [] # hidden outputs from Transform for each AE (total n_ae = 5)
    for n in range(5):
        H_ec_n = Reshape([seq_length, 20])(in_enc_connection[:,n,:])
        H_ec_n = SimpleRNN(16)(H_ec_n)
        H_ec.append(H_ec_n)
    H_ec = Concatenate()(H_ec)
    
    H_dc = [] # hidden outputs from Transform for each AE
    for n in range(5):
        H_dc_n = Reshape([seq_length, 20])(in_dec_connection[:,n,:])
        H_dc_n = SimpleRNN(16)(H_dc_n)
        H_dc.append(H_dc_n)
    H_dc = Concatenate()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")              
    
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")     
    outputs = decoder(H)
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_CRAE")
    return model, encoder, decoder    


def PASTA_CLAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    
    seq_length, n_heads, ff_dim, dropout = connection_shape[0][1], 8, z_dim, 0.1
    
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])

    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_network_ops) # channel ==> encoder/decoder
    H_n = MaxPool2D(3, padding = 'same')(H_n)
    H_n = Conv2D(1, 1, activation = 'relu')(H_n)
    H_n = Flatten()(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_temp_ops) # channel ==> encoder/decoder
    H_t = MaxPool2D(3, padding = 'same')(H_t)
    H_t = Conv2D(1, 1, activation = 'relu')(H_t)
    H_t = Flatten()(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)    

    H_ec = [] # hidden outputs from Transform for each AE (total n_ae = 5)
    for n in range(5):
        H_ec_n = Reshape([seq_length, 20])(in_enc_connection[:,n,:])
        H_ec_n = LSTM(16)(H_ec_n)
        H_ec.append(H_ec_n)
    H_ec = Concatenate()(H_ec)
    
    H_dc = [] # hidden outputs from Transform for each AE
    for n in range(5):
        H_dc_n = Reshape([seq_length, 20])(in_dec_connection[:,n,:])
        H_dc_n = LSTM(16)(H_dc_n)
        H_dc.append(H_dc_n)
    H_dc = Concatenate()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")              
    
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")     
    outputs = decoder(H)
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_CLAE")
    return model, encoder, decoder


def PASTA_CGAE(setting_shape, connection_shape, z_dim:int, features:str):
    tf.keras.backend.clear_session()
    
    seq_length, n_heads, ff_dim, dropout = connection_shape[0][1], 8, z_dim, 0.1
    
    in_setting = Input(shape = setting_shape[0]) # anom. scoring fn., etc.
    in_network_ops = Input(shape = setting_shape[1]) # layer type, etc.
    in_temp_ops = Input(shape = setting_shape[2]) # temporal connectivity types
    in_enc_connection = Input(shape = connection_shape[0])
    in_dec_connection = Input(shape= connection_shape[1])

    H_g = Dense(16, activation = 'relu')(in_setting)
    
    H_n = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_network_ops) # channel ==> encoder/decoder
    H_n = MaxPool2D(3, padding = 'same')(H_n)
    H_n = Conv2D(1, 1, activation = 'relu')(H_n)
    H_n = Flatten()(H_n)
    H_n = Dense(16, activation = 'relu')(H_n)
    
    H_t = Conv2D(16, 2, activation = 'relu', data_format = 'channels_first')(in_temp_ops) # channel ==> encoder/decoder
    H_t = MaxPool2D(3, padding = 'same')(H_t)
    H_t = Conv2D(1, 1, activation = 'relu')(H_t)
    H_t = Flatten()(H_t)
    H_t = Dense(16, activation = 'relu')(H_t)    

    H_ec = [] # hidden outputs from Transform for each AE (total n_ae = 5)
    for n in range(5):
        H_ec_n = Reshape([seq_length, 20])(in_enc_connection[:,n,:])
        H_ec_n = GRU(16)(H_ec_n)
        H_ec.append(H_ec_n)
    H_ec = Concatenate()(H_ec)
    
    H_dc = [] # hidden outputs from Transform for each AE
    for n in range(5):
        H_dc_n = Reshape([seq_length, 20])(in_dec_connection[:,n,:])
        H_dc_n = GRU(16)(H_dc_n)
        H_dc.append(H_dc_n)
    H_dc = Concatenate()(H_dc)
    
    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        H = Concatenate()([H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")        
    elif features == 'no_net':
        H = Concatenate()([H_g, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_enc_connection, in_dec_connection], outputs = H, name="encoder")
    elif features == 'no_temp':
        H = Concatenate()([H_g, H_n, H_t])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops], outputs = H, name="encoder")          
    else:
        H = Concatenate()([H_g, H_n, H_t, H_ec, H_dc])
        H = Dense(z_dim, name='encoder_output')(H)
        encoder = Model(inputs = [in_setting, in_network_ops, in_temp_ops, in_enc_connection, in_dec_connection], outputs = H, name="encoder")              
    
    in_decoder = Input(shape = [z_dim])
    X = Dense(z_dim, activation = 'relu')(in_decoder)
    
    O_g = Dense(16, activation = 'relu')(X) # positive outputs (i.e., settings/configs)
    O_g = Dense(math.prod(setting_shape[0]), activation = 'relu')(O_g) # output only 0 or 1 as onehot vectors
    O_g = Reshape(list(setting_shape[0]))(O_g)
    O_g = Softmax()(O_g)
    
    O_n = Dense(16, activation = 'relu')(X)
    O_n = Dense(math.prod(setting_shape[1]), activation = 'relu')(O_n)
    O_n = Reshape(setting_shape[1])(O_n)
    O_n = Softmax()(O_n)
    
    O_t = Dense(16, activation = 'relu')(X)
    O_t = Dense(math.prod(setting_shape[2]), activation = 'relu')(O_t)
    O_t = Reshape(setting_shape[2])(O_t)
    O_t = Softmax()(O_t)
    
    O_ec = Dense(16, activation='relu')(X)
    O_ec = Dense(math.prod(connection_shape[0]))(O_ec) # output only 0 or negative values
    O_ec = ReLU(negative_slope = 1., max_value = 1.)(O_ec) # negative outputs (i.e., temporal connections)
    O_ec = Reshape(connection_shape[0])(O_ec)
    
    O_dc = Dense(16, activation='relu')(X)
    O_dc = Dense(math.prod(connection_shape[1]))(O_dc) # output only 0 or negative values
    O_dc = ReLU(negative_slope = 1., max_value = 1.)(O_dc) # negative outputs (i.e., temporal connections)
    O_dc = Reshape(connection_shape[1])(O_dc)

    # 'all', 'no_task', 'no_net', 'no_temp'       
    if features == 'no_task':
        decoder = Model(inputs = in_decoder, outputs = [O_n, O_t, O_ec, O_dc], name="decoder")
    elif features == 'no_net':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_ec, O_dc], name="decoder")
    elif features == 'no_temp':
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t], name="decoder")        
    else:
        decoder = Model(inputs = in_decoder, outputs = [O_g, O_n, O_t, O_ec, O_dc], name="decoder")     
    outputs = decoder(H)
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name="PASTA_CGAE")
    return model, encoder, decoder