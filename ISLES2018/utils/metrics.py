from functools import partial

from keras import backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy
#from keras.losses import category_crossentropy
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


#def binary_focal_loss(gamma=2.0, alpha=0.25):
def focal_loss(y_true, y_pred):
    gamma=5.0
    alpha=0.75
   # epsilon = 0.00001
   # y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
   # cross_entropy = -y_true * K.log(y_pred)
   # weight = alpha * y_true * K.pow((1 - y_pred), gamma)
   # loss = weight * cross_entropy
   # loss = K.sum(loss, axis=(-3, -2, -1))
   # return loss
    epsilon = K.epsilon()
    print("ZY", epsilon)
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1 - p_t), gamma)
    loss = weight * cross_entropy
    loss = K.mean(loss, axis=1)
    return loss

   # return focal_loss

def weighted_focal_loss(y_true, y_pred):
    return -focal_loss(y_true, y_pred)



def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    r= K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))
   # f= ((binary_crossentropy(y_true, y_pred)) + smooth/2)
  #  h=r + f
    return tf.reshape(r, (-1, 1, 1))

def weighted_dice_coefficient_loss(y_true, y_pred):
    return -(weighted_dice_coefficient(y_true, y_pred) - binary_crossentropy(y_true, y_pred))


#def binary_crossentropy(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001): 
 #   return K.mean((K.binary_crossentropy(y_true, y_pred)) + smooth/2)




#def binary_cross_entropy_loss(y_true, y_pred):
 #   return -binary_crossentropy(y_true, y_pred)


#total_loss = binary_cross_entropy_loss(y_true, y_pred) + weighted_dice_coefficient_loss(y_true, y_pred)



#def weighted_binary_crossentropy(y_true, y_pred):
 #   return 


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
