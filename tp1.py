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

    denoiseWindowName = 'Denoise Window'
    denoiseWindowTb = 'Denoise Window TB'
    createWindowWithTrackbar(denoiseWindowName, denoiseWindowTb, 1, 7) # No pongas 0 crashea

    key = 'a'

    while key != ord('z'):
        # 1 - Get original image
        _, originalImage = webcam.read()
        originalImage = cv.flip(originalImage, 1) # espejamos para que se vea bien
        # cv.imshow('Original image', originalImage)

        # 2 - Get binary image
        binaryValue = cv.getTrackbarPos(trackbarName, windowName)
        binaryImage = getBinaryImage(originalImage, binaryValue)
        cv.imshow(windowName, binaryImage)

        # 3 - Remove noise
        radius = cv.getTrackbarPos(denoiseWindowTb, denoiseWindowName)
        denoisedImage = denoiseImage(binaryImage, radius)

        cv.imshow(denoiseWindowName, denoisedImage)

        # 4 - Contours
        contours, hierarchy = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cv.drawContours(originalImage, contours, -1, (100, 0, 100), 2)
        cv.imshow("Contours Window", originalImage) #muestra contornos violetas pelados
        #convex_hull(contours, originalImage) # muestra contornos con convexHull

        key = cv.waitKey(30)

def convex_hull(contours, originalImage):
    hull = []
    for cnt in contours:
        hull.append(cv.convexHull(cnt, False))
    cv.drawContours(originalImage, hull, -1, (255, 0, 0), 3)
    cv.imshow("Contours", originalImage)

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