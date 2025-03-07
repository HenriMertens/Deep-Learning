import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img




# Get current working directory (where the script is located)
base_dir = os.getcwd()

# Construct the correct path
test_dir = os.path.join(base_dir, "simpsons-mnist-master", "dataset", "rgb", "test")
zoom_in_dir = os.path.join(base_dir, "simpsons-mnist-master", "dataset", "rgb", "test_zoomed_in")
zoom_out_dir = os.path.join(base_dir, "simpsons-mnist-master", "dataset", "rgb", "test_zoomed_out")



# Create output directories if they don't exist
os.makedirs(zoom_in_dir, exist_ok=True)
os.makedirs(zoom_out_dir, exist_ok=True)

# Create zoomed-in and zoomed-out generators
zoom_in_gen = ImageDataGenerator(zoom_range=[1.2, 1.5])  # Zooms IN (1.2x to 1.5x)
zoom_out_gen = ImageDataGenerator(zoom_range=[0.7, 0.9])  # Zooms OUT (0.7x to 0.9x)

# Loop through each character folder
for character in os.listdir(test_dir):
    character_path = os.path.join(test_dir, character)
    zoom_in_character_path = os.path.join(zoom_in_dir, character)
    zoom_out_character_path = os.path.join(zoom_out_dir, character)

    # Ensure it's a directory and create corresponding output folders
    if os.path.isdir(character_path):
        os.makedirs(zoom_in_character_path, exist_ok=True)
        os.makedirs(zoom_out_character_path, exist_ok=True)

        for img_name in os.listdir(character_path):
            img_path = os.path.join(character_path, img_name)

            # Load image
            img = load_img(img_path)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Apply zoom-in augmentation
            for batch in zoom_in_gen.flow(img_array, batch_size=1):
                zoomed_in_img = array_to_img(batch[0])
                zoomed_in_img.save(os.path.join(zoom_in_character_path, img_name))
                break  # Only save one variation per image

            # Apply zoom-out augmentation
            for batch in zoom_out_gen.flow(img_array, batch_size=1):
                zoomed_out_img = array_to_img(batch[0])
                zoomed_out_img.save(os.path.join(zoom_out_character_path, img_name))
                break  # Only save one variation per image

print("All images zoomed in and out successfully!")
