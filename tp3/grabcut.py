import cv2 as cv
from pip._vendor.distlib.compat import raw_input
import numpy as np
import variables


def main():
    grabcut(captureImage())

def captureImage():
    raw_input('Press ENTER to capture the image')
    camera = cv.VideoCapture(variables.webcamId)
    _, image = camera.read()
    image = cv.flip(image, 1)
    return image

def grabcut(image):
    mask = np.zeros(image.shape[:2], np.uint8)

    # These are arrays used by the algorithm internally.
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Usamos selectROI para seleccionar el rectangulo
    rectangle = cv.selectROI("Select frame", image, fromCenter=False, showCrosshair=True)

    cv.grabCut(image, mask, rectangle, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    image = image * mask2[:, :, np.newaxis]

    cv.imshow("Output with the mask", image)
    cv.waitKey()


main()
cv.destroyAllWindows()