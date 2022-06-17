import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('Capture.jpg')  # positioning images
imgVideo = cv2.imread('Capture.jpg')  # showing images

# success, imgVideo = myVid.read()  # debug
hT, wT, cT = imgTarget.shape  # get the height(hT), width(wT), color(cT) variables from imgTarget
imgVideo = cv2.resize(imgVideo, (wT, hT))  # resize imgVideo to imgTarget

orb = cv2.ORB_create(nfeatures=2000)  # nfeatures=2000 can have the best positioning
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)  # debug

while True:  # loop until user choose to exit
    # sample code view https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)  # debug

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    # print(len(good))  # debug
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if len(good) > 30:  # if 30 out of 2000 points matched then generate imgVideo
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, [255, 0, 255], 3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

        cv2.imshow('imgAug', imgAug)
    # cv2.imshow('imgWarp', imgWarp)
    # cv2.imshow('img2', img2)
    cv2.imshow('imgFeatures', imgFeatures)
    # cv2.imshow('imgTarget', imgTarget)
    # cv2.imshow('myVid', imgVideo)
    # cv2.imshow('imgWebcam', imgWebcam)
    cv2.waitKey(1)
