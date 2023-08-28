
import cv2 as cv
import variables


def main():
    shapePrecisionThreshold = 0

    webcam = cv.VideoCapture(variables.videoCaptureId)

    # Window + trackbar creation
    windowName = 'Tp1'
    trackbarName = 'Trackbar'
    createWindowWithTrackbar(windowName, trackbarName)

    denoiseWindowName = 'Denoise Window'
    denoiseWindowTb = 'Denoise Window TB'
    createWindowWithTrackbar(denoiseWindowName, denoiseWindowTb, 1, 7) # No pongas 0 crashea

    # creamos la window con trackbar de precision (cuanto mayor valor, mayor dificultad - mas precisa debe ser la forma)
    createWindowWithTrackbar('contornos', 'Precision')

    # window original con la trackbar que regula el tamaño acpetado de las figuras a tener en cuenta
    createWindowWithTrackbar('OriginalImage', 'TamanioContorno', 0, 100000)

    key = 'a'

    while key != ord('z'):

        # 1 - LECTURA DEL VALOR DE LOS TRACKBARS DE LAS 4 WINDOWS   
        shapePrecisionThreshold = cv.getTrackbarPos('Precision', 'contornos') #obtiene el valor de precision de trackbar 
        binaryValue = cv.getTrackbarPos(trackbarName, windowName)
        radius = cv.getTrackbarPos(denoiseWindowTb, denoiseWindowName)
        shapeContourSize = cv.getTrackbarPos('TamanioContorno', 'OriginalImage')

        #USO DE LOS VALORES DE LOS TRACKBARS DE LAS 4 WINDOWS
        # 2 - Get los contour prototypes (figuras de ejemplo + valor de los threshold de precision)
            # como yo uso windows necesito cambiar las barras al reves que si no no funciona 
            #lo metemos dentro del while para que la variable de precision de la trackbar se actualice constantemente
        contourAndContourNames = [
            (getContoursByImage(variables.squareImagePath, shapePrecisionThreshold), 'Square'), 
            (getContoursByImage(variables.triangleImagePath, shapePrecisionThreshold), 'Triangle'),
            (getContoursByImage(variables.circleImagePath, shapePrecisionThreshold), 'Circle'),
        ]

        # 3 - Get original image
        _, originalImage = webcam.read()
        originalImage = cv.flip(originalImage, 1) # espejamos para que se vea bien

        # 4 - Get binary image
        binaryImage = getBinaryImage(originalImage, binaryValue)
        cv.imshow(windowName, binaryImage) # Required

        # 5 - Remove noise
        denoisedImage = denoiseImage(binaryImage, radius) 
        cv.imshow(denoiseWindowName, denoisedImage) # Required

        # 6 - Get Contours
        contours, hierarchy = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # 7 - COMPROBAMOS QUE DETECTE BIEN LOS CONTORNOS (debugear)
        orignialImageCopy = originalImage

        # 8 - Filter and compare contours
        for contour in contours:
            if cv.contourArea(contour) > shapeContourSize: # Checks that contour size is big enough fot taking into account
                allDefinedShapesInvalid = True
                for contourShape, contourName in contourAndContourNames:
                    if doesContourMatchShapesContour(contourShape, contour):
                        allDefinedShapesInvalid = False
                        displayValidShape(contour, contourName, originalImage)
                        break
                if allDefinedShapesInvalid:
                    displayInvalidShape(contour, originalImage)

            cv.imshow('OriginalImage', originalImage)

        key = cv.waitKey(30)

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

def createWindowWithTrackbar(windowName, trackbarName, initRange = 0, endRange = 255):
    cv.namedWindow(windowName, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(windowName, 600, 337) #pongo esto porq en mi pc las windows se veian enorme y era re incomodo
    cv.createTrackbar(trackbarName, windowName, initRange, endRange, (lambda a: None)) #investigar porq el valor del tercer parametro no influye en el valor minimo del trackbar

def createWindow(windowName):
    cv.namedWindow(windowName, cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(windowName, 800, 450) #pongo esto porq en mi pc las windows se veian enorme y era re incomodo
    

main()
cv.destroyAllWindows()
