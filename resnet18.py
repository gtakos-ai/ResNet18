def resnetLayer(inputs,
            out_filters=64,
            in_filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
            batchNormalization=True,
            residual=True):

    conv1 = Conv2D(out_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    conv2 = Conv2D(out_filters,
                  kernel_size=kernel_size,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if batchNormalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    x = conv1(x)
    if batchNormalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    x = conv2(x)
    if residual:
        x = keras.layers.add([x, inputs])
    return x

def resnetBlock(inputs,
            in_filters=64,
            out_filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
            batchNormalization=True):


    x = inputs
    for i in range(3):
        print("Layer = ", i)
        if (i==0):
            res=False
        else:
            res=True
            strides=1
        x = resnetLayer(inputs=x,
            in_filters=in_filters,
            out_filters=out_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            batchNormalization=batchNormalization,
            residual=res)
    return x

def firstLayer(inputs, output_filters=64):
    conv = Conv2D(output_filters, (3,3), padding='same')
    x = conv(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def resNet18(input_shape, numClasses=10):
    output_filters = 16
    depth = 4 # 4*depth + 2 layers
    inputs = Input(shape=input_shape)
    x = firstLayer(inputs, output_filters=output_filters)
    for i in range(depth):
        print("Depth = ", i)
        if (i>0):
            strides=2
        else:
            strides=1
        in_filters = output_filters*strides
        x = resnetBlock(inputs=x, 
            in_filters=in_filters, 
            out_filters=output_filters,
            kernel_size=3,
            strides=strides,
            activation='relu',
            batchNormalization=True)
        output_filters*=2
        
    poolSize = int(input_shape[1]/(2**(depth-1)))   
    x = AveragePooling2D(pool_size=(poolSize,poolSize))(x)
    
    y = Flatten()(x)
    outputs = Dense(numClasses,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
