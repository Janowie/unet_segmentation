import tensorflow as tf
import keras


def conv_relu_block(input_tensor, num_of_filters, batch_norm):

    t = keras.layers.Conv2D(filters=num_of_filters,
                                    kernel_size=(3, 3),
                                    kernel_initializer = 'he_normal',
                                    strides=1,
                                    padding='same',
                                    data_format='channels_first')(input_tensor)
    if batch_norm:
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
    t = keras.layers.Activation('relu')(t)
    
    t = keras.layers.Conv2D(filters=num_of_filters,
                                    kernel_size=(3, 3),
                                    kernel_initializer = 'he_normal',
                                    strides=1,
                                    padding='same',
                                    data_format='channels_first')(t)
    if batch_norm:
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
#     us = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    concat = keras.layers.Concatenate(axis=1)(inputs=[us, encode_input_tensor])
    return concat


def validate_conf(conf):
    validated_conf = {
        "filters": [1, 64, 128, 256, 512, 1024],
        "dropout": 0.05,
        "batch_norm": True,
        "last_layer_activation": "sigmoid"
    }
    for key in conf.keys():
        if key in validated_conf:
            validated_conf[key] = conf[key]
    return validated_conf


def UNet(channels, image_size, conf):
    
    validated_conf = validate_conf(conf)

    filters = validated_conf['filters']
    input_data = keras.layers.Input(shape=(channels, image_size, image_size))

    # encode
    cr1 = conv_relu_block(input_data, filters[1], validated_conf["batch_norm"])
    if validated_conf["dropout"]:
        cr1 = keras.layers.Dropout(validated_conf["dropout"])(cr1)
    md1 = max_pool(cr1)
    
    cr2 = conv_relu_block(md1, filters[2], validated_conf["batch_norm"])
    if validated_conf["dropout"]:
        cr2 = keras.layers.Dropout(validated_conf["dropout"])(cr2)
    md2 = max_pool(cr2)
    
    cr3 = conv_relu_block(md2, filters[3], validated_conf["batch_norm"])
    if validated_conf["dropout"]:
        cr3 = keras.layers.Dropout(validated_conf["dropout"])(cr3)
    md3 = max_pool(cr3)
    
    cr4 = conv_relu_block(md3, filters[4], validated_conf["batch_norm"])
    if validated_conf["dropout"]:
        cr4 = keras.layers.Dropout(validated_conf["dropout"])(cr4)
    md4 = max_pool(cr4)

    # bottom
    cr5 = conv_relu_block(md4, filters[5], validated_conf["batch_norm"])

    # decode
    us1 = up_sample(cr5, cr4)
    if validated_conf["dropout"]:
        us1 = keras.layers.Dropout(validated_conf["dropout"])(us1)
    cr6 = conv_relu_block(us1, filters[4], validated_conf["batch_norm"])
    
    us2 = up_sample(cr6, cr3)
    if validated_conf["dropout"]:
        us2 = keras.layers.Dropout(validated_conf["dropout"])(us2)
    cr7 = conv_relu_block(us2, filters[3], validated_conf["batch_norm"])
    
    us3 = up_sample(cr7, cr2)
    if validated_conf["dropout"]:
        us3 = keras.layers.Dropout(validated_conf["dropout"])(us3)
    cr8 = conv_relu_block(us3, filters[2], validated_conf["batch_norm"])
    
    us4 = up_sample(cr8, cr1)
    if validated_conf["dropout"]:
        us4 = keras.layers.Dropout(validated_conf["dropout"])(us4)
    cr9 = conv_relu_block(us4, filters[1], validated_conf["batch_norm"])

    output_data = keras.layers.Conv2D(filters=filters[0],
                                      kernel_size=(1, 1),
                                      padding="same",
                                      activation=validated_conf["last_layer_activation"],
                                      data_format='channels_first')(cr9)

    model = keras.models.Model(input_data,
                               output_data,
                               name="UNet")

    return model