import os
import random
import cv2 # type: ignore


# Get current working directory (where the script is located)
base_dir = os.getcwd()

# Construct the correct path
test_dir = os.path.join(base_dir, "simpsons-mnist-master", "dataset", "rgb", "test")
rotated_dir = os.path.join(base_dir, "simpsons-mnist-master", "dataset", "rgb", "test_rotated")



# Define possible rotations (in degrees)
rotations = [90, 180, 270]

# Create the new directory if it doesn't exist
os.makedirs(rotated_dir, exist_ok=True)

# Loop through each character folder
for character in os.listdir(test_dir):
    character_path = os.path.join(test_dir, character)
    rotated_character_path = os.path.join(rotated_dir, character)

    # Ensure it's a directory and create corresponding folder in test_rotated
    if os.path.isdir(character_path):
        os.makedirs(rotated_character_path, exist_ok=True)

        for img_name in os.listdir(character_path):
            img_path = os.path.join(character_path, img_name)
            rotated_img_path = os.path.join(rotated_character_path, img_name)

            # Read image
            img = cv2.imread(img_path)

            if img is not None:
                # Choose a random rotation
                angle = random.choice(rotations)

                # Apply rotation
                if angle == 90:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == 180:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                elif angle == 270:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # Save rotated image to test_rotated directory
                cv2.imwrite(rotated_img_path, rotated_img)

print("All images rotated and saved in 'test_rotated' successfully!")
