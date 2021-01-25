import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

######################################################################

H = 0
S = 1
W = 1
V = 2
L = 2
B = 2

critCond = {
    "saturation": 35,
    "colorH": 330,
    "colorL": 65,
    "lightness": 70,
    "colorIrrelW": 98,
    "colorIrrelB": 2
}

blobMinSize = 150

######################################################################

# Load the image and converts BGR to RGB
def loadImage(filePath):
    imBGR = cv2.imread(filePath)
    return imBGR


# Image blurring by convolving image with low-pass filter, removes noise - 7x7 kernel
def blurImage(img):
    return cv2.blur(img,(7,7))


def convertHSVToHWB(imgHSV, imgShape):
    height, width, x = imgShape
    imgHWB = imgHSV

    for i in range(height):
        for j in range(width):
            imgHWB[i,j][H] = imgHSV[i,j][0]                         # H - H | H = H
            imgHWB[i,j][W] = (1 - imgHSV[i,j][S]) * imgHSV[i,j][V]  # W - S | W = (1 - S)V
            imgHWB[i,j][B] = 1 - imgHSV[i,j][V]                     # B - V | B = 1 - V

    return imgHWB


# Converts the RGB image to HSV, HLS and HWB
def convertImages(blurImgBGR, imgShape):
    imgHSV = cv2.cvtColor(blurImgBGR, cv2.COLOR_BGR2HSV)
    imgHLS = cv2.cvtColor(blurImgBGR, cv2.COLOR_BGR2HLS)
    imgHWB = convertHSVToHWB(imgHSV, imgShape)

    return imgHSV, imgHLS, imgHWB


def calculateAvg(img, variable):
    height, width, x = img.shape

    total = 0
    count = 0

    for i in range(height):
        for j in range(width):
            count += 1
            total += img[i,j][variable]

    return round(total/count)


def imageSegmentation(convertedImg, imgShape):
    height, width, x = imgShape
    hybridImage = np.zeros((height, width))

    imgHSV, imgHLS, imgHWB = convertedImg
    avgB = calculateAvg(imgHWB, B)
    avgW = calculateAvg(imgHWB, W)
    avgS = calculateAvg(imgHSV, S)

    for i in range(height):
        for j in range(width):
            pixelHSV = imgHSV[i,j]
            pixelHLS = imgHLS[i,j]
            pixelHWB = imgHWB[i,j]

            cond1 = ((pixelHSV[S] > critCond["saturation"]) and (pixelHSV[H] >= critCond["colorH"] and pixelHSV[H] <= critCond["colorL"]) and (pixelHLS[L] > critCond["lightness"]))
            cond2 = (pixelHWB[B] < avgB and pixelHWB[W] < avgW)
            cond3 = (pixelHWB[W] > avgW and pixelHSV[S] > avgS)
            cond4 = (pixelHWB[W] >= critCond["colorIrrelW"] and pixelHWB[B] <= critCond["colorIrrelB"])

            if (cond1 or cond2 or cond3 or cond4) == True:
                hybridImage[i,j] = 255
            else:
                hybridImage[i,j] = 0

    return hybridImage


def filterBlobs(hybridImage, imgShape):

    hybridImage = (hybridImage).astype('uint8')
    height, width, x = imgShape

    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor=True
    params.blobColor=255
    params.filterByArea = True
    params.minArea = 0
    params.maxArea = blobMinSize
    params.minDistBetweenBlobs = 0

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(hybridImage)
    for point in keypoints:
        y = round(point.pt[0])
        x = round(point.pt[1])
        hybridImage[x,y] = 0

    return hybridImage


######################################################################

def plotImages(hybridImage, img):
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.subplot(122),plt.imshow(hybridImage, cmap='Greys',  interpolation='nearest'),plt.title('Hybrid - fire shown white')
    plt.show()

######################################################################

def processImage(filePath):
    imBGR = loadImage(filePath)
    blurredImgBGR = blurImage(imBGR)

    convertedImg = convertImages(blurredImgBGR, imBGR.shape)

    hybridImage = imageSegmentation(convertedImg, imBGR.shape)
    return filterBlobs(hybridImage, imBGR.shape), imBGR

######################################################################

def main():
    for file in os.listdir("images"):
        filePath = os.path.join("images", file)
        hybridImage, img = processImage(filePath)
        plotImages(hybridImage, img)


if __name__ == "__main__":
    main()