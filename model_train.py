
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from keras.utils import Sequence
from keras.callbacks import ReduceLROnPlateau
from keras.metrics import Precision, Recall, AUC
from keras.optimizers import Adam

from utils.model import Unet
from utils.losses import dice_score, dice_loss
from utils.preprocess import preprocess
from utils.utils import masks_as_image, get_image

# Set logging level to ERROR to suppress warnings
tf.get_logger().setLevel('ERROR')

# Configuration settings for the model and training
class CFG:
    img_size = (768, 768, 3)
    mask_size = (768, 768, 1)
    train_bs = 64
    valid_bs = 2 * train_bs
    valid_size = 0.3
    n_classes = 1
    n_epochs = 5
    min_lr = 1e-7
    min_delta = 1e-4
    train_folder = "./ship_detection_data/train_v2/"
    csv_path = "./ship_detection_data/train_ship_segmentations_v2.csv"

# Custom data generator class for creating batches
class DataGenerator(Sequence):
    def __init__(self, dataframe, batch_size):
        super().__init__()
        self.dataframe = dataframe
        self.unique_df = dataframe.drop_duplicates('ImageId')
        self.batch_size = batch_size

    # Compute the number of batches in the dataset
    def __len__(self):
        return np.ceil(len(self.dataframe) / self.batch_size)
    
    # Generate one batch of data
    def __getitem__(self, index):

        low = index * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.unique_df))

        batch_df = self.unique_df.iloc[low : high]

        if len(batch_df) == 0:
        # Handle the case when batch_df is empty (e.g., at the end of the dataset)
            return (
                np.zeros((self.batch_size, *CFG.img_size)),
                np.zeros((self.batch_size, *CFG.mask_size))
            )
        # Load and preprocess images and masks for the current batch
        return (np.array([get_image(img_path)
                          for img_path in batch_df['ImageId'].values]),
                np.array([masks_as_image(self.dataframe[self.dataframe['ImageId'] == img_path]['EncodedPixels'])
                          for img_path in batch_df['ImageId'].values]))
    
def main():
    # Preprocess training and validation data
    train_df, valid_df = preprocess(path=CFG.csv_path, valid_size=CFG.valid_size)

    # Create data generators for training and validation
    train_gen = DataGenerator(train_df, CFG.train_bs)
    valid_gen = DataGenerator(valid_df, CFG.valid_bs)

    # Create U-Net like model
    model = Unet(CFG.img_size)

    # Compile the model with optimizer, loss function, and evaluation metrics
    model.compile(optimizer=Adam(),
                  loss=dice_loss,
                  metrics=[dice_score,
                           Precision(),
                           Recall(),
                           AUC()])
    
    # Set up a learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=.5,
                                     patience=3, verbose=1, mode='max',
                                     min_delta=CFG.min_delta, cooldown=2,
                                     min_lr=CFG.min_lr)
    
    # Train the model
    model.fit(train_gen,
              validation_data=valid_gen,
              epochs=CFG.n_epochs,
              callbacks=[reduce_lr])
    
    # Check if save folder exists
    save_folder = 'models'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the model
    model.save(f'{save_folder}/model_diceloss.h5')

if __name__ == "__main__":
    main()