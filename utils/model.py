import tensorflow.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input,\
      Activation, concatenate
from keras.models import Model

def conv2d_block(input_tensor, n_filters, kernel_size = 3):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x = Activation('relu')(x)
    
    return x

def Unet(input_img, n_filters = 16):
    """Function to define the U-Net Model"""

    classes = 1
    # We use a slight deviation on the U-Net standart

    # Input layer
    input = Input(shape=input_img)

    # Contracting Path
    c1 = conv2d_block(input, n_filters * 1, kernel_size = 3)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # U-Net bottom
    c4 = conv2d_block(p3, n_filters = n_filters * 8, kernel_size = 3)
    
    # Expansive Path
    u5 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u5 = concatenate([u5, c3])
    c5 = conv2d_block(u5, n_filters * 4, kernel_size = 3)
    
    u6 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c2])
    c6 = conv2d_block(u6, n_filters * 2, kernel_size = 3)
    
    u7 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c1])
    c7 = conv2d_block(u7, n_filters * 1, kernel_size = 3)
    
    # Output Layer
    output = Conv2D(classes, (1, 1), activation='sigmoid')(c7)

    # Generate Keras model
    model = Model(inputs=input, outputs=output)

    return model
