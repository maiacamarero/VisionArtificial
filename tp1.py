import cv2 as cv


def main():
    # circleToCompare = getContoursByShape('./tp1/555.png', 100)
    # squareToCompare = getContoursByShape('./tp1/square3.png', 100)
    # triangleToCompare = getContoursByShape('./tp1/triangle.png', 100)

    webcam = cv.VideoCapture(0)

    # Window + trackbar creation
    windowName = 'Tp1'
    trackbarName = 'Trackbar'
    createWindowWithTrackbar(windowName, trackbarName)

    denoiseWindow = 'Denoise Window'
    denoiseWindowTb = 'Denoise Window TB'
    createWindowWithTrackbar(denoiseWindow, denoiseWindowTb, 1, 7) # No pongas 0 crashea

    key = 'a'

    while key != ord('z'):
        # 1 - Get original image
        _, originalImage = webcam.read()
        # cv.imshow('Original image', originalImage)

        # 2 - Get binary image
        binaryValue = cv.getTrackbarPos(trackbarName, windowName)
        binaryImage = getBinaryImage(originalImage, binaryValue)
        cv.imshow(windowName, binaryImage)

        # 3 - Remove noise
        radius = cv.getTrackbarPos(denoiseWindowTb, denoiseWindow)
        denoisedImage = denoiseImage(binaryImage, radius)
        
        cv.imshow(denoiseWindow, denoisedImage)


        key = cv.waitKey(30)

def denoiseImage(binaryImage, radius):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))  # kernel = structural element shape in which 
    opening = cv.morphologyEx(binaryImage, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

def getBinaryImage(image, value):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret1, thresh = cv.threshold(grayImage, value, 255, cv.THRESH_BINARY)
    return thresh


def getContoursByShape(image_route, thresh_bottom):
    shape = cv.imread(image_route)
    grayShape = cv.cvtColor(shape, cv.COLOR_BGR2GRAY)
    ret, shapeThresh = cv.threshold(grayShape, thresh_bottom, 255, cv.THRESH_BINARY_INV)
    shapeContours, hierarchy = cv.findContours(shapeThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image=shape, contours=shapeContours, contourIdx=-1, color=(255, 255, 0), thickness=3)
    return shapeContours[0]

def createWindowWithTrackbar(windowName, trackbarName, initRange = 0, endRange = 255):
    cv.namedWindow(windowName)
    cv.createTrackbar(trackbarName, windowName, initRange, endRange, (lambda a: None))

main()
cv.destroyAllWindows()