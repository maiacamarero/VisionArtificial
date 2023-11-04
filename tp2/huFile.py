import os
import cv2 as cv
import pandas as pd

#------------------------------------------

import variables

#------------------------------------------

def calculate_hu_moments(image_path):
    # Load the image in grayscale
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Calculate Hu moments
    moments = cv.moments(image)
    hu_moments = cv.HuMoments(moments).flatten()

    # Convert NumPy array to a list of floats
    hu_moments_list = [float(moment) for moment in hu_moments]

    return hu_moments_list

#------------------------------------------

def process_images_folder():
    base_folder = variables.base_folder_path

    # Initialize an empty list to store the results
    hu_moments_data = []

    # Iterate over subfolders (classes)
    for label, class_folder in enumerate(os.listdir(base_folder), start=1):
        class_folder_path = os.path.join(base_folder, class_folder)

        # Iterate over images in the class folder
        for filename in os.listdir(class_folder_path):
            if filename.endswith(".png"):
                image_path = os.path.join(class_folder_path, filename)
                hu_moments_list = calculate_hu_moments(image_path)

                # Append label and Hu moments to the list
                hu_moments_data.append({'label': label, 'hu_moments': hu_moments_list})

    # Create a DataFrame from the list
    hu_moments_df = pd.DataFrame(hu_moments_data)

    # Save the DataFrame to a CSV file
    hu_moments_df.to_csv("extra_huMoments.csv", index=False)
    print("HuMoments have been saved.")

#------------------------------------------

if __name__ == "__main__":
    process_images_folder()