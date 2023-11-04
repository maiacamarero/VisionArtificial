import cv2 as cv
import joblib

#------------------------------------------

import huMomentsGenerator
import variables
import labels

# -----------------------------------------

def calculate_hu_moments(contour):
    # Ensure the contour is a valid NumPy array
    if len(contour) == 0:
        return None

    # Ensure the contour is a 2D array
    if len(contour.shape) != 2:
        contour = contour.squeeze()

    # Calculate Hu moments from the contour
    moments = cv.moments(contour)
    hu_moments = cv.HuMoments(moments).flatten()

    # Convert NumPy array to a list of floats
    hu_moments_list = [float(moment) for moment in hu_moments]

    return hu_moments_list

# -----------------------------------------

def main():

    # -----------------------------------------

    webcam = cv.VideoCapture(variables.webcamId)

    window_name = 'Image Processing'
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(window_name, 1200, 674)

    # Denoise Trackbar
    denoise_trackbar_name = 'Denoise Level'
    cv.createTrackbar(denoise_trackbar_name, window_name, 1, 7, (lambda a: None))

    # Binary Trackbar
    binary_trackbar_name = 'Binary Threshold'
    cv.createTrackbar(binary_trackbar_name, window_name, 0, 255, (lambda a: None))

    # Original Image Trackbar
    contour_size_trackbar_name = 'Contour Size'
    cv.createTrackbar(contour_size_trackbar_name, window_name, 2500, 50000, (lambda a: None))

    # -----------------------------------------

    while True:
        # -----------------------------------------

        binary_value = cv.getTrackbarPos(binary_trackbar_name, window_name)
        radius = cv.getTrackbarPos(denoise_trackbar_name, window_name)

        _, original_image = webcam.read()
        original_image = cv.flip(original_image, 1)

        binary_image = huMomentsGenerator.get_binary_image(original_image, binary_value)
        denoised_image = huMomentsGenerator.denoise_image(binary_image, radius)
        cv.imshow(window_name, denoised_image)

        contours, _ = cv.findContours(denoised_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        second_largest_contour = huMomentsGenerator.get_second_largest_contour(contours)

        # -----------------------------------------

        if second_largest_contour is not None:
            cv.drawContours(original_image, [second_largest_contour], -1, (255, 0, 255), 3)
            cv.imshow(window_name, original_image)

            # -----------------------------------------

            key = cv.waitKey(10)

            # Save Hu Moments when 'k' is pressed
            if key == ord('k'):
                
                # ----------------------------------------- 
                # Implementando el trainer para predecir la clasificacion de la nueva figura

                knn_super_model = joblib.load("knn_super_model.joblib")

                hu_moments = calculate_hu_moments(second_largest_contour)

                predicted_label = knn_super_model.predict([hu_moments])[0]
                predicted_label_str = labels.int_to_label(predicted_label)
                print(f"Predicted Label: {predicted_label} ({predicted_label_str})")
    
                # -----------------------------------------

                labeled_image = original_image.copy()
                cv.putText(labeled_image, f"Predicted Label: {predicted_label} ({predicted_label_str})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow("Labeled Contour", labeled_image)

            # -----------------------------------------

            elif key == ord("x"):
                break

    # -----------------------------------------

    cv.destroyAllWindows()

# -----------------------------------------

if __name__ == "__main__":
    main()