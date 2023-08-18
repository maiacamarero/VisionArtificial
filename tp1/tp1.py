import cv2 as cv


def main():
    contourAndContourNames = [
        (getContoursByImage('./square.png', 100), 'Square'),
        (getContoursByImage('./triangle.png', 100), 'Triangle'),
        (getContoursByImage('./circle.png', 100), 'Circle'),
    ]

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

        # 2 - Get binary image
        binaryValue = cv.getTrackbarPos(trackbarName, windowName)
        binaryImage = getBinaryImage(originalImage, binaryValue)
        cv.imshow(windowName, binaryImage) # Required

        # 3 - Remove noise
        radius = cv.getTrackbarPos(denoiseWindowTb, denoiseWindowName)
        denoisedImage = denoiseImage(binaryImage, radius) 
        cv.imshow(denoiseWindowName, denoisedImage) # Required

        # 4 - Contours
        contours, hierarchy = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # 6 - Filter and compare contours
        for contour in contours:
            if cv.contourArea(contour) > 10000: # Checks that contour size is big enough
                for contourShape, contourName in contourAndContourNames:
                    if doesContourMatchShapesContour(contourShape, contour):
                        displayValidShape(contour, contourName, originalImage)
                        break

        cv.imshow('Original Image', originalImage)

        key = cv.waitKey(30)

def doesContourMatchShapesContour(circleContour, contour):
    return cv.matchShapes(circleContour, contour, cv.CONTOURS_MATCH_I2, 0) < 0.03

def displayValidShape(contour, shapeName, originalImage):
    x, y, _, _ = cv.boundingRect(contour)
    cv.putText(originalImage, shapeName, (x, y), cv.FONT_ITALIC, 4, (255, 0, 127), 1, cv.LINE_4)
    cv.drawContours(originalImage, contour, -1, (255, 0, 127), 3)

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


def getContoursByImage(image_route, thresh_bottom):
    shape = cv.imread(image_route)
    grayShape = cv.cvtColor(shape, cv.COLOR_BGR2GRAY)
    ret, shapeThresh = cv.threshold(grayShape, thresh_bottom, 255, cv.THRESH_BINARY_INV)
    shapeContours, hierarchy = cv.findContours(shapeThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(shape, shapeContours, -1, (255, 255, 0), 3)
    return shapeContours[0]

def createWindowWithTrackbar(windowName, trackbarName, initRange = 0, endRange = 255):
    cv.namedWindow(windowName)
    cv.createTrackbar(trackbarName, windowName, initRange, endRange, (lambda a: None))


main()
cv.destroyAllWindows()
