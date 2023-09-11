
import cv2 as cv
import variables
import functools

def main():

    webcam = cv.VideoCapture(variables.webcamId)

    # Window + trackbar creation
    denoiseWindowName = 'Binary Image'
    cv.namedWindow(denoiseWindowName, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(denoiseWindowName, 600, 337) #pongo esto porq en mi pc las windows se veian enorme y era re incomodo

    denoiseWindowTb = 'Noise'
    cv.createTrackbar(denoiseWindowTb, denoiseWindowName, 1, 7, (lambda a: None)) #investigar porq el valor del tercer parametro no influye en el valor minimo del trackbar
    
    binaryTrackbarName = 'Binary'
    cv.createTrackbar(binaryTrackbarName, denoiseWindowName, 0, 255, (lambda a: None)) #investigar porq el valor del tercer parametro no influye en el valor minimo del trackbar


    # window original con la trackbar que regula el tamaño acpetado de las figuras a tener en cuenta
    originalImageWindowName='Original Image'
    cv.namedWindow(originalImageWindowName, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(originalImageWindowName, 600, 337)
    
    key = 'a'

    while key != ord('z'):

        # LECTURA DEL VALOR DE LOS TRACKBARS DE LAS 4 WINDOWS   
        binaryValue = cv.getTrackbarPos(binaryTrackbarName, denoiseWindowName)
        radius = cv.getTrackbarPos(denoiseWindowTb, denoiseWindowName)

        # Get original image
        _, originalImage = webcam.read()
        originalImage = cv.flip(originalImage, 1) # espejamos para que se vea bien

        # Get binary image
        binaryImage = getBinaryImage(originalImage, binaryValue)

        # Remove noise
        denoisedImage = denoiseImage(binaryImage, radius) 
        cv.imshow(denoiseWindowName, denoisedImage) # Required

        # Get Contours
        contours, _ = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        getBiggestContour(contours)
        
        # Show image with contours
        cv.imshow(originalImageWindowName, originalImage)

        key = cv.waitKey(30)

def getBiggestContour(contours):
    return functools.reduce(lambda a, b: a if cv.contourArea(a) > cv.contourArea(b) else b, contours)

def doesContourMatchShapesContour(contourShape, contour):
    return cv.matchShapes(contourShape, contour, 1, 0.0) < 0.03

def displayInvalidShape(contour, originalImage):
    x, y, _, _ = cv.boundingRect(contour)
    cv.putText(originalImage, "Invalid", (x, y), cv.FONT_ITALIC, 1.5, (255, 255, 255), 1, cv.LINE_4)
    cv.drawContours(originalImage, contour, -1, (0, 0, 255), 3)

def displayValidShape(contour, shapeName, originalImage):
    x, y, _, _ = cv.boundingRect(contour)
    cv.putText(originalImage, shapeName, (x, y), cv.FONT_ITALIC, 1.5, (255, 255, 255), 1, cv.LINE_4)
    cv.drawContours(originalImage, contour, -1, (0, 255, 0), 3)

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
    return shapeContours[0]


main()
cv.destroyAllWindows()
