import os
import csv
import cv2 as cv

#------------------------------------------

import variables_toni

#------------------------------------------

def get_binary_image(image, value):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_image, value, 255, cv.THRESH_BINARY)
    return thresh

def denoise_image(binary_image, radius):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
    opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

def get_second_largest_contour(contours):
    if len(contours) > 1:
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        return sorted_contours[1]
    return None

def save_hu_moments(contour, label):
    # Check if the file exists, if not, write the header
    header_exists = os.path.exists('huMoments.csv')
    with open('huMoments.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if it doesn't exist
        if not header_exists:
            writer.writerow(['label', 'hu_moments'])

        # Calculate and save Hu Moments
        hu_moments = cv.HuMoments(cv.moments(contour)).flatten()
        formatted_hu_moments = ', '.join(f'{val}' for val in hu_moments)
        writer.writerow([label, f'[{formatted_hu_moments}]'])
        print("HuMoments with label {} have been saved.".format(label))

#------------------------------------------

def main():

    #------------------------------------------

    webcam = cv.VideoCapture(variables_toni.webcamId)

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
    
    #------------------------------------------
    
    while True:
        #------------------------------------------

        binary_value = cv.getTrackbarPos(binary_trackbar_name, window_name)
        radius = cv.getTrackbarPos(denoise_trackbar_name, window_name)

        _, original_image = webcam.read()
        original_image = cv.flip(original_image, 1)

        binary_image = get_binary_image(original_image, binary_value)
        denoised_image = denoise_image(binary_image, radius)
        cv.imshow(window_name, denoised_image)

        contours, _ = cv.findContours(denoised_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        second_largest_contour = get_second_largest_contour(contours)

        #------------------------------------------

        if second_largest_contour is not None:
            cv.drawContours(original_image, [second_largest_contour], -1, (255, 0, 255), 3)
            cv.imshow(window_name, original_image)

            #------------------------------------------

            key = cv.waitKey(10)

            if key == ord('k'):
                label = input("Enter the label (1, 2, 3, etc.): ")
                save_hu_moments(second_largest_contour, label)

            #------------------------------------------
                
            elif key == ord('x'):
                break

    #------------------------------------------

    cv.destroyAllWindows()

#------------------------------------------

if __name__ == "__main__":
    main()