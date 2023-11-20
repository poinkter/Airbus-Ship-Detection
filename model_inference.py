import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.model import Unet  
from utils.utils import multi_rle_encode, get_image

# Configuration settings for the model inference
class CFG:
    img_size = (768, 768, 3)
    max_images = 10 # Max number of images to display
    model_path = './models/model_diceloss.h5'
    test_path = './ship_detection_data/test_v2'  # Change to your test images path
    save_path = './ship_detection_data/'

def load_model(model_path):
    """
    Load U-Net model
    """
    model = Unet(CFG.img_size)
    model.load_model(model_path)
    return model

def display_images(original_images, predicted_images):
    """
    Display original images and predicted masks by the model side by side.
    """

    num_images = min(len(original_images), len(predicted_images), CFG.max_images)

    plt.figure(figsize=(4 * num_images, 8))

    for i in range(num_images):
        # Original Image
        plt.subplot(num_images, 2,  2 * i + 1)
        plt.imshow(original_images[i])
        plt.title('Original image')
        plt.axis('off')

        # Predicted Image
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(predicted_images[i])
        plt.title('Predicted Mask')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def predict_masks(display_images=True):
    """
    Predict masks using a Keras model and optionally display the images and predicted masks.
    """
    # Load the trained model
    model = load_model(CFG.model_path)

    # Load a sample image for inference
    sample_images = np.array([get_image(CFG.test_path + img_path)
                             for img_path in os.listdir(CFG.test_path)])
    
    # Perform inference
    predictions = model.predict(sample_images)

    # Show predicted masks
    if display_images:
        display_images(sample_images, predictions)

    # Convert the predicted images to RLE encoding
    prediction_rows = []
    for prediction, img_path in zip(predictions, os.listdir(CFG.test_path)):
        encodings = multi_rle_encode(prediction)
        # Add an entry with np.NaN if there is no ship detected
        prediction_rows.append([{'ImageId': img_path, 'EncodedPixels': encoding} 
                                if encodings
                                else {'ImageId': img_path, 'EncodedPixels': np.NaN} 
                                for encoding in encodings])
    
    return pd.DataFrame(prediction_rows)


def main():
    submission_df = predict_masks(display_images=False)
    # Save the results
    submission_df.to_csv(CFG.save_path + 'submission.csv', index=False)

if __name__ == "__main__":
    main()