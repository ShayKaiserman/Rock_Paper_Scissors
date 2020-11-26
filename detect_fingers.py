import cv2 as cv
import numpy as np

# ----- functions ------

# function that subscribe the first image(img1) from the second image (img2)
def background_sub(img1,img2):
    # we assume that the images are at the same size
    return img1-img2

def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        cnt = max(contours, key=lambda x: cv.contourArea(x))
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        cv.drawContours(img, cnt, -1, (255, 0, 0), 3)
    return cnt

# ----- script -----

cap = cv.VideoCapture(0)
frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)
start = False

# text output of player move
text = "None"

while True:

    ret, frame = cap.read()
    k = cv.waitKey(10)

    img = frame[:,:]

    if k == ord(' '):
        img_background = frame
        print("background saved")
        start = True

    if start:
        img = background_sub(frame,img_background)
        frame_blur = cv.GaussianBlur(img, (5, 5), 0)
        img_gray = cv.cvtColor(frame_blur, cv.COLOR_BGR2GRAY)
        ret, img_bin = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
        img_edge = cv.Canny(img_gray, 100, 200)
        cv.imshow("Blur", frame_blur)
        cv.imshow("Edges", img_bin)
        print(img_bin.shape)
        print(img_bin.dtype)
        print(img_edge.shape)

        contours, hierarchy = cv.findContours(img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if k == ord("s"):
        # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cont = max(contours, key=lambda x: cv.contourArea(x))
        cv.drawContours(img, [cont], -1, (255, 255, 0), 2)
        cv.imshow("contours", img)
        # hull = cv.convexHull(contours)
        hull = cv.convexHull(cont)
        cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
        cv.imshow("hull", img)
        # hull = cv.convexHull(contours, returnPoints=False)
        hull = cv.convexHull(cont, returnPoints=False)
        defects = cv.convexityDefects(cont, hull)

        cnt = 0
        if defects is not None:
            # cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(cont[s][0])
                end = tuple(cont[e][0])
                far = tuple(cont[f][0])
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv.circle(img, far, 4, [0, 0, 255], -1)

        if cnt > 2:
            text = "paper"
        elif cnt == 2:
            text = "scissors"
        else:
            text = "rock"

        cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(img, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("hull", img)

    cv.imshow('final_result', img)

    if k == ord("q"):
        break
