from keras.models import load_model
import cv2
import numpy as np
import sys
import os

# ----- functions ------

def mapper(val):
    return REV_CLASS_MAP[val]

# function that subscribe the first image(img1) from the second image (img2)
def background_sub(img1,img2):
    # we assume that the images are at the same size
    return img1-img2

# function that choose the computer move according the player predicted move
def cp_move(player_move):
    if player_move == "rock":
        computer_move = "paper"
    if player_move == "paper":
        computer_move = "scissors"
    if player_move == "scissors":
        computer_move = "rock"
    return computer_move

# function to detect fingers and count them
def det_fingers(img):
    # default test
    text = "rock"

    # image pre-processing
    frame_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    cv2.imshow("Edges", img_bin)

    # find contours
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is not None:
        cont = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(img, [cont], -1, (255, 255, 0), 2)
        cv2.imshow("contours", img)

        # for display
        hull = cv2.convexHull(cont)
        cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)
        # for use
        hull = cv2.convexHull(cont, returnPoints=False)

        # counting the defects
        defects = cv2.convexityDefects(cont, hull)
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
                    cv2.circle(img, far, 4, [0, 0, 255], -1)

        if cnt == 0:
            text = "rock"
        elif cnt == 1:
            text = "scissors"
        else:
            text = "paper"

        cv2.putText(img, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("hull", img)

    return text

# ----- script ------

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "None"
}

# model = load_model("rock-paper-scissors-model.h5")
model = load_model("rock-paper-scissors-model-2.h5")

cap = cv2.VideoCapture(0)
frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)
start = False # changes if we want to capture images

print("capturing video...")
print("press 'b' to capture background image")

while True:
    ret, frame = cap.read()

    k = cv2.waitKey(1)

    # save the background image (the reference to the subtraction)
    if k == ord('b'):
        # img_background = frame[100:400, 100:400]
        img_background = frame[:,:]
        print("background saved")

    if start:
        # take an image
        img = frame[:,:]
        # background subtraction
        img_sub = background_sub(img, img_background)
        # resize the image to the original training input of MobileNet
        img_new = cv2.resize(img_sub, (227, 227),  interpolation=cv2.INTER_AREA)
        img_new = np.where(img_new > 128, 255, 0)
        # print(img_new.shape)

        # predict the player move
        pred = model.predict(np.array([img_new]))
        print(pred)
        move_code = np.argmax(pred[0])
        player_move = mapper(move_code)
        # print("Predicted: {}".format(move_name))

        if player_move == "None":
            cv2.putText(frame, "Predicted: {}".format(player_move),
                    (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # compute the computer move and display it
        if player_move != "None":
            player_move = det_fingers(img_sub)
            computer_move = cp_move(player_move)
            cv2.putText(frame, "Predicted: {}".format(player_move),
                        (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Computer Move: {}".format(computer_move),
                        (300, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            icon = cv2.imread(
                "images/{}.png".format(computer_move))
            icon = cv2.resize(icon, (200, 200))
            frame[100:300, 400:600] = icon

    cv2.imshow("Video", frame)

    if k == ord(' '):
        start = not start

    if k == ord('q'):
        break




