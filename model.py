import tensorflow as tf
import keras


def conv_relu_block(input_tensor, num_of_filters):

    t = keras.layers.Conv2D(filters=num_of_filters,
                                    kernel_size=(3, 3),
                                    kernel_initializer = 'he_normal',
                                    strides=1,
                                    padding='same',
                                    data_format='channels_first')(input_tensor)
    t = keras.layers.BatchNormalization(axis=1, 
                                    momentum=0.99, 
                                    epsilon=0.001, 
                                    center=True, 
                                    scale=True, 
                                    beta_initializer='zeros', 
                                    gamma_initializer='ones', 
                                    moving_mean_initializer='zeros', 
                                    moving_variance_initializer='ones', 
                                    beta_regularizer=None, 
                                    gamma_regularizer=None, 
                                    beta_constraint=None, 
                                    gamma_constraint=None)(t)
#     t = tfa.layers.InstanceNormalization(axis=1)(t)
    t = keras.layers.Activation('relu')(t)
    
    t = keras.layers.Conv2D(filters=num_of_filters,
                                    kernel_size=(3, 3),
                                    kernel_initializer = 'he_normal',
                                    strides=1,
                                    padding='same',
                                    data_format='channels_first')(t)
    t = keras.layers.BatchNormalization(axis=1, 
                                    momentum=0.99, 
                                    epsilon=0.001, 
                                    center=True, 
                                    scale=True, 
                                    beta_initializer='zeros', 
                                    gamma_initializer='ones', 
                                    moving_mean_initializer='zeros', 
                                    moving_variance_initializer='ones', 
                                    beta_regularizer=None, 
                                    gamma_regularizer=None, 
                                    beta_constraint=None, 
                                    gamma_constraint=None)(t)
#     t = tfa.layers.InstanceNormalization(axis=1)(t)
    t = keras.layers.Activation('relu')(t)

    return t


def max_pool(input_tensor):
    return keras.layers.MaxPooling2D(pool_size=(2, 2),
                                     strides=None,
                                     padding='same',
                                     data_format='channels_first')(input_tensor)


def up_sample(input_tensor, encode_input_tensor):
    us = keras.layers.UpSampling2D(size=(2, 2),
                                   data_format='channels_first')(input_tensor)
    concat = keras.layers.Concatenate(axis=1)(inputs=[us, encode_input_tensor])
    return concat


def UNet(channels, image_size):

    filters = [1, 64, 128, 256, 512, 1024]

    input_data = keras.layers.Input(shape=(channels, image_size, image_size))

    # encode
    cr1 = conv_relu_block(input_data, filters[1])
#     cr1 = keras.layers.Dropout(0.25)(cr1) # test dropout
    md1 = max_pool(cr1)
    
    cr2 = conv_relu_block(md1, filters[2])
#     cr2 = keras.layers.Dropout(0.35)(cr2) # test dropout
    md2 = max_pool(cr2)
    
    cr3 = conv_relu_block(md2, filters[3])
#     cr3 = keras.layers.Dropout(0.4)(cr3) # test dropout
    md3 = max_pool(cr3)
    
    cr4 = conv_relu_block(md3, filters[4])
#     cr4 = keras.layers.Dropout(0.2)(cr4) # test dropout
    md4 = max_pool(cr4)

    # bottom
    cr5 = conv_relu_block(md4, filters[5])
#     cr5 = conv_relu_block(md3, filters[4])

    # decode
    us1 = up_sample(cr5, cr4)
#     us1 = keras.layers.Dropout(0.2)(us1) # test dropout
    cr6 = conv_relu_block(us1, filters[4])
    
    us2 = up_sample(cr6, cr3)
#     us2 = keras.layers.Dropout(0.4)(us2) # test dropout
    cr7 = conv_relu_block(us2, filters[3])
    
    us3 = up_sample(cr7, cr2)
#     us3 = keras.layers.Dropout(0.35)(us3) # test dropout
    cr8 = conv_relu_block(us3, filters[2])
    
    us4 = up_sample(cr8, cr1)
#     us4 = keras.layers.Dropout(0.25)(us4) # test dropout
    cr9 = conv_relu_block(us4, filters[1])

    output_data = keras.layers.Conv2D(filters=filters[0],
                                      kernel_size=(1, 1),
                                      padding="same",
                                      activation="sigmoid",
                                      data_format='channels_first')(cr9)

    model = keras.models.Model(input_data,
                               output_data,
                               name="UNet")
    
    print("Compiled with input shape", input_data.shape)

    return model