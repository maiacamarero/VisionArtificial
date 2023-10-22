import csv

import cv2 as cv
import numpy

import variables
import labels


def main():
    webcam = cv.VideoCapture(variables.webcamId)

    # Window + trackbar creation
    denoiseWindowName = 'Binary Image'
    cv.namedWindow(denoiseWindowName, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(denoiseWindowName, 600, 337)

    denoiseWindowTb = 'Noise'
    cv.createTrackbar(denoiseWindowTb, denoiseWindowName, 1, 7, (lambda a: None))

    binaryTrackbarName = 'Binary'
    cv.createTrackbar(binaryTrackbarName, denoiseWindowName, 0, 255, (lambda a: None))

    # window original con la trackbar que regula el tamaño acpetado de las figuras a tener en cuenta
    originalImageWindowName = 'Original Image'
    cv.namedWindow(originalImageWindowName, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(originalImageWindowName, 600, 337)

    contourSizeTrackbarName = 'Contour size'
    cv.createTrackbar(contourSizeTrackbarName, originalImageWindowName, 2500, 10000, (lambda a: None))

    precisionTrackbarName = 'Precision'
    cv.createTrackbar(precisionTrackbarName, originalImageWindowName, 0, 255, (lambda a: None))

    key = 'a'

    label, description = labels

    while key != ord('z'):

        # 1 - LECTURA DEL VALOR DE LOS TRACKBARS DE LAS 4 WINDOWS
        binaryValue = cv.getTrackbarPos(binaryTrackbarName, denoiseWindowName)
        radius = cv.getTrackbarPos(denoiseWindowTb, denoiseWindowName)

        # 3 - Get original image
        _, originalImage = webcam.read()
        originalImage = cv.flip(originalImage, 1)  # espejamos para que se vea bien

        # 4 - Get binary image
        binaryImage = getBinaryImage(originalImage, binaryValue)

        # 5 - Remove noise
        denoisedImage = denoiseImage(binaryImage, radius)
        cv.imshow(denoiseWindowName, denoisedImage)  # Required

        # 6 - Get Contours
        contours, _ = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            biggestContour = contours[0]
            for cnt in contours:
                if cv.contourArea(cnt) > cv.contourArea(biggestContour):
                    biggestContour = cnt
            cv.drawContours(originalImage, biggestContour, -1,	(255, 0, 0), 3)

        # Show image with contours
        cv.imshow(originalImageWindowName, originalImage)

        if cv.waitKey(1) & 0xFF == ord('k'):
            if biggestContour is not None:
                with open('huMoments.csv', 'a', '') as file:
                    writer = csv.writer(file)
                    huMoments = cv.HuMoments(cv.moments(biggestContour))
                    writer.writerow(numpy.append(int(label), huMoments))
                    print("Saved")

        key = cv.waitKey(30)

def denoiseImage(binaryImage, radius):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))  # kernel = structural element shape in which
    opening = cv.morphologyEx(binaryImage, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)


def getBinaryImage(image, value):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret1, thresh = cv.threshold(grayImage, value, 255, cv.THRESH_BINARY)
    return thresh


main()
cv.destroyAllWindows()
