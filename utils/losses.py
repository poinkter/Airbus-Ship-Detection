import keras.backend as K

def dice_score(y_true, y_pred, smooth=1.):
    """
    Callculate the Dice score between the ground truth and predicted arrays

    Dice = (2*|X & Y|) / (|X| + |Y|) =
         = 2*sum(|A*B|) / (sum(A^2) + sum(B^2))
    """
    # Ensure consistent data types
    y_true = K.cast(y_true, dtype='float32')  
    y_pred = K.cast(y_pred, dtype='float32')

    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(K.square(y_true)) + K.sum(K.square(y_pred))
    return (2. * intersection) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)
