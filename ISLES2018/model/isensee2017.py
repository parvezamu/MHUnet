from functools import partial
#import tensorflow as tf
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss, dice_coefficient
#from keras_diagram import ascii
from keras_sequential_ascii import keras2ascii

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


# create_separable_convolution = partial(create_separable_convolution, activation=LeakyReLU, instance_normalization=True)


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.2,
                      n_segmentation_levels=4, n_labels=4, optimizer=Adam, initial_learning_rate=3e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid",
                      metrics=dice_coefficient):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)
    dilation_rate = (2, 2, 2)
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
#        if (level_number == 4):
 #           n_level_filters = 256
  #          level_filters.append(n_level_filters)
   #     elif (level_number < 4):
        n_level_filters = (2 ** level_number) * n_base_filters
            # if(level_number > 1):
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv1 = create_convolution_block(current_layer, n_level_filters)
            #            in_conva=create_convolution_block(in_conv1, n_level_filters, strides=(2, 2, 2))
            # in_convb = create_convolution_block(in_conv1, n_level_filters)
            # in_convc = create_convolution_block(in_convb, n_level_filters)
            in_conv2 = create_context_module2(in_conv1, n_level_filters, dilation_rate, dropout_rate=dropout_rate)
            in_conv2 = concatenate([in_conv1, in_conv2], axis=1)
            in_conv = create_convolution_block(in_conv2, n_level_filters, kernel=(1, 1, 1))
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))
        #     if(level_number<2):
        #        if(level_number==1):
        #       for i in range(2):
        #               if(i==0):
        #             context_output_layer1 = create_context_module(in_conv, n_level_filters,  dropout_rate=dropout_rate)
        #       context_output_layer1=Add([in_conv, context_output_layer1])
        #
        #      elif(i==1):
        #     #     context_output_layer = create_context_module(context_output_layer1, n_level_filters,
        #                                                dropout_rate=dropout_rate)
        #          context_output_layer = Add([context_output_layer1, context_output_layer])
        #         summation_layer = concatenate([context_output_layer1, context_output_layer])
        #         level_output_layers.append(summation_layer)
        #    current_layer = summation_layer
        #   else:
        #            context_output_layer= create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)
        #        summation_layer = concatenate([in_conv, context_output_layer], axis=1)
        #         level_output_layers.append(summation_layer)
        #      current_layer = summation_layer
        if (level_number < 5):
            for i in range(1):
                if (i == 0):
                    context_output_layer = create_context_module1(in_conv, n_level_filters,
                                                                  dropout_rate=dropout_rate)
                    summation_layer = concatenate([in_conv, context_output_layer], axis=1)
                    level_output_layers.append(summation_layer)
                    current_layer = summation_layer
                    print("Current_layer", current_layer)
    #  elif(level_number==4):
    #      for i in range(1):
    #         if (i == 0):
    #            context_output_layer = create_context_module2(in_conv, n_level_filters, dilation_rate,  dropout_rate=dropout_rate)
    #           summation_layer = concatenate([in_conv, context_output_layer], axis=1)
    #          level_output_layers.append(summation_layer)
    #         current_layer = summation_layer
    # elif (i == 1):
    #   context_output_layer2 = create_context_module2(context_output_layer1, n_level_filters, dilation_rate,
    #           dropout_rate=dropout_rate)
    #   elif (i == 2):#
    #        context_output_layer3 = create_context_module2(context_output_layer2, n_level_filters, dilation_rate,  dropout_rate=dropout_rate)
    #    elif (i == 3):
    #       context_output_layer = create_context_module2(context_output_layer3, n_level_filters, dilation_rate,
    #        dropout_rate=dropout_rate)
    #   if (i == 3):
    #      summation_layer = Add()([in_conv, context_output_layer])
    #      level_output_layers.append(summation_layer)
    #   current_layer = summation_layer
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling1 = create_up_sampling_module(current_layer, level_filters[level_number], dilation_rate,
                                                 dropout_rate=dropout_rate)
        up_sampling = create_convolution_block(up_sampling1, 2 * level_filters[level_number], kernel=(1, 1, 1))
        if (level_number == 3):

            a1 = create_convolution_block(current_layer, level_filters[level_number + 1], kernel=(1, 1, 1))
            a2 = UpSampling3D(size=(2, 2, 2))(a1)
            # a3=create_convolution_block(a1, level_filters[3], kernel=(1, 1, 1))
            a4 = Add()([up_sampling, a2])
            print("A", a4)
            ################################################################
            current_layer0 = level_output_layers[3]
            aa0 = level_filters[3]
            aa1 = create_convolution_block(current_layer0, 2 * aa0, kernel=(1, 1, 1))
            aa2 = Add()([up_sampling, aa1])
            ##############################################################################
            current_layer1 = level_output_layers[2]
            # print("A5", current_layer1)
            aa = level_filters[2]
            # print("D", aa)
            a5 = create_convolution_block(current_layer1, aa, kernel=(1, 1, 1))
            a6 = create_convolution_block(a5, aa, strides=(2, 2, 2))
            a7 = create_convolution_block(a6, 4 * aa, kernel=(1, 1, 1))
            a8 = Add()([up_sampling, a7])
            print("B", a8)
            ########################################################################
            current_layer2 = level_output_layers[1]
            ab = level_filters[1]
            a9 = create_convolution_block(current_layer2, ab, kernel=(1, 1, 1))
            a10 = create_convolution_block(a9, ab, strides=(4, 4, 4))
            a11 = create_convolution_block(a10, 8 * ab, kernel=(1, 1, 1))
            a12 = Add()([up_sampling, a11])
            print("C", a12)
            #################################################################################
            current_layer3 = level_output_layers[0]
            ac = level_filters[0]
            a13 = create_convolution_block(current_layer3, ac, kernel=(1, 1, 1))
            a14 = create_convolution_block(a13, ac, strides=(8, 8, 8))
            a15 = create_convolution_block(a14, 16 * ac, kernel=(1, 1, 1))
            a16 = Add()([up_sampling, a15])
            print("D", a16)
            ################################################################################
            a17 = Add()([up_sampling, aa2, a4, a8, a12, a16])
            print("E", a17)
            up_sampling2 = create_up_sampling_module1(a17, 2 * level_filters[level_number], dilation_rate,
                                                      dropout_rate=dropout_rate)
            print("F", up_sampling2)
            concatenation_layer = Add()([level_output_layers[level_number], up_sampling2])
            print("G", concatenation_layer)
        else:
            concatenation_layer = Add()([level_output_layers[level_number], up_sampling])
        localization_output1 = create_localization_module(concatenation_layer, level_filters[level_number],
                                                          dropout_rate=dropout_rate)
        localization_output = create_convolution_block(localization_output1, level_filters[level_number],
                                                       kernel=(1, 1, 1))
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    # output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if (level_number == 3):
            segmentation_layer = UpSampling3D(size=(8, 8, 8))(segmentation_layer)
            output_layer0 = segmentation_layers[0]
            output_layer0 = Add()([output_layer0, segmentation_layer])


        elif (level_number == 2):
            segmentation_layer = UpSampling3D(size=(4, 4, 4))(segmentation_layer)
            output_layer2 = segmentation_layers[0]
            output_layer2 = Add()([output_layer2, segmentation_layer, output_layer0])
        elif (level_number == 1):
            segmentation_layer = UpSampling3D(size=(2, 2, 2))(segmentation_layer)
            output_layer3 = segmentation_layers[0]
            output_layer3 = Add()([output_layer3, segmentation_layer, output_layer2, output_layer0])
        elif (level_number == 0):
            output_layer4 = segmentation_layers[0]
            output_layer = Add()([output_layer0, output_layer2, output_layer3, output_layer4])

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    # parallel_model = multi_gpu_model(model, gpus=2)
    print(model.summary())
    print(keras2ascii(model))
#    print(ascii(model))

    if not isinstance(metrics, list):
        metrics = [metrics]
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
    return model


def create_localization_module(input_layer, n_filters, dropout_rate=0.2, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer, n_filters=2)
    convolution1 = concatenate([input_layer, convolution1], axis=1)
    dropout = SpatialDropout3D(rate=0.2, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(dropout, n_filters=2)
    convolution2 = concatenate([dropout, convolution2], axis=1)
    convolution3 = create_convolution_block(convolution2, n_filters=2)
    convolution3 = concatenate([convolution2, convolution3], axis=1)
    convolution4 = create_convolution_block(convolution3, n_filters, kernel=(1, 1, 1))
    return convolution4


def create_up_sampling_module(input_layer, n_filters, dilation_rate, size=(2, 2, 2), dropout_rate=0.2,
                              data_format="channels_first"):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution1 = create_convolution_block(up_sample, n_filters, kernel=(1, 1, 1))
    convolution2 = create_convolution_block(convolution1, n_filters=2)
    convolution2 = concatenate([convolution1, convolution2], axis=1)
    dropout = SpatialDropout3D(rate=0.2, data_format=data_format)(convolution2)
    convolution3 = create_convolution_block(dropout, n_filters=2, dilation_rate=dilation_rate)
    convolution3 = concatenate([dropout, convolution3], axis=1)
    convolution4 = create_convolution_block(dropout, n_filters=2, dilation_rate=(3, 3, 3))
    convolution4 = concatenate([dropout, convolution4], axis=1)
    con=concatenate([convolution3, convolution4], axis=1)
    con1=create_convolution_block(con, n_filters, kernel=(1, 1, 1))
    dropout1=create_convolution_block(dropout, n_filters, kernel=(1, 1, 1))
    concat = Add()([dropout1, con1])
    convolution6 = create_convolution_block(concat, n_filters, kernel=(1, 1, 1))
    # convolution7=create_convolution_block(convolution6, n_filters)
    return convolution6


def create_context_module(input_layer, n_level_filters, dropout_rate=0.2, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2


# def create_context_module1(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
#    convolutionxy=create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
#   convolutionxy=concatenate([input_layer, convolutionxy], axis=1)
#   dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolutionxy)
#   convolution1 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters, kernel=(1, 3, 3))
#  convolutiona=create_convolution_block(input_layer=convolution1, n_filters=n_level_filters, kernel=(3, 1, 1))
#  concat1=concatenate([convolution1, convolutiona, dropout], axis=1)
#  convolution2=create_convolution_block(input_layer=concat1, n_filters=n_level_filters, kernel=(3, 3, 1))
# convolutionb=create_convolution_block(input_layer=convolution2, n_filters=n_level_filters, kernel=(1, 1, 3))
#   concat2=concatenate([convolution1, convolutiona, convolution2, convolutionb, dropout], axis=1)
#  convolution3=create_convolution_block(input_layer=concat2, n_filters=n_level_filters, kernel=(3, 1, 3))
#  convolutionc=create_convolution_block(input_layer=convolution3, n_filters=n_level_filters, kernel=(1, 3, 1))
#   concat3=concatenate([convolutiona, convolutionb, convolutionc], axis=1)
# convolution7=create_convolution_block(input_layer=concat3, n_filters=n_level_filters, kernel=(1, 1, 1))
#  return convolution7

def create_context_module1(input_layer, n_level_filters, dropout_rate=0.2, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=2)
    convolution1 = concatenate([input_layer, convolution1], axis=1)
    dropout = SpatialDropout3D(rate=0.2, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=2)
    convolution2 = concatenate([dropout, convolution1], axis=1)
    convolution3 = create_convolution_block(input_layer=convolution2, n_filters=2)
    convolution3 = concatenate([convolution2, convolution3], axis=1)
    convolution7 = create_convolution_block(input_layer=convolution3, n_filters=n_level_filters, kernel=(1, 1, 1))
    return convolution7


def create_context_module2(input_layer, n_level_filters, dilation_rate, dropout_rate=0.2, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=2, dilation_rate=dilation_rate)
    convolution2 = concatenate([dropout, convolution2], axis=1)
    convolution3 = create_convolution_block(input_layer=dropout, n_filters=2, dilation_rate=(3, 3, 3))
    convolution3 = concatenate([dropout, convolution3], axis=1)
    convolution4 = create_convolution_block(input_layer=dropout, n_filters=2, dilation_rate=(5, 5, 5))
    convolution4 = concatenate([dropout, convolution4], axis=1)
    concat1 = concatenate([convolution2, convolution3, convolution4], axis=1)
    con11 = create_convolution_block(concat1, n_filters=n_level_filters, kernel=(1, 1, 1))
    dropout1=create_convolution_block(dropout, n_filters=n_level_filters, kernel=(1, 1, 1))
    concat2 = Add()([dropout1, con11])
    convolution6 = create_convolution_block(input_layer=concat2, n_filters=n_level_filters, kernel=(1, 1, 1))
    return convolution6


def create_up_sampling_module1(input_layer, n_filters, dilation_rate, dropout_rate=0.2, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer, n_filters, kernel=(1, 1, 1))
    dropout = SpatialDropout3D(rate=0.2, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(dropout, n_filters=2)
    convolution3 = create_convolution_block(dropout, n_filters=2, dilation_rate=dilation_rate)
    concat1 = concatenate([convolution2, convolution3], axis=1)
    concat11 = concatenate([dropout, convolution2], axis=1)
    concate12 = concatenate([dropout, convolution3], axis=1)
    #############################################################################################
    convolution4 = create_convolution_block(concat11, n_filters=2, dilation_rate=dilation_rate)
    convolution5 = create_convolution_block(concate12, n_filters=2, dilation_rate=(3, 3, 3))
    concat2 = concatenate([convolution4, convolution5], axis=1)
    concat21 = concatenate([dropout, concat11, convolution4], axis=1)
    concat22 = concatenate([dropout, concate12, convolution5], axis=1)
    ##############################################################################################
    convolution6 = create_convolution_block(concat21, n_filters=2, dilation_rate=(3, 3, 3))
    convolution7 = create_convolution_block(concat22, n_filters=2, dilation_rate=dilation_rate)
    concat3 = concatenate([convolution6, convolution7], axis=1)
    # concat31=concatenate([dropout, convolution2, convolution4, convolution6], axis=1)
    # concat32=concatenate([dropout, convolution3, convolution5, convolution7], axis=1)
    ################################################################################################
#    concata = concatenate([dropout, concat1, concat2, convolution6], axis=1)
 #   concatb = concatenate([dropout, concat2, concat3, convolution7], axis=1)
  #  concatc = concatenate([concata, concatb], axis=1)
   # dropout1 = SpatialDropout3D(rate=0.2, data_format=data_format)(concatc)
    ######################################################################################################
    #convolution8 = create_convolution_block(dropout1, n_filters=2, dilation_rate=dilation_rate)
    #convolution9 = create_convolution_block(dropout1, n_filters=2, dilation_rate=(3, 3, 3))
   # concat3 = concatenate([convolution8, convolution9], axis=1)
    ###############################################################################################
    concat = concatenate([dropout, concat1, concat2, concat3], axis=1)
    convolution10 = create_convolution_block(concat, n_filters, kernel=(1, 1, 1))
    # convolution7=create_convolution_block(convolution6, n_filters)
    return convolution10
