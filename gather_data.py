import os, os.path
import cv2
import numpy as np

# ----- functions ------

# function that subscribe the first image(img1) from the second image (img2)
def background_sub(img1,img2):
    # we assume that the images are at the same size
    return img1-img2

# function that count files in given directory (folder), and print it
def CountFiles(DIR):
    c = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    print('Number of images in the folder: ', c)
    return c

# ----- script ------

#  user choose a label
label_name = input("insert a label: ")

# define path to save the images

PATH = os.getcwd()+'\\collected data\\train'
SAVE_PATH = os.path.join(PATH, label_name)

try:
    os.mkdir(SAVE_PATH)
    N_files = CountFiles(SAVE_PATH)
except FileExistsError:
    print("{} directory already exists.".format(SAVE_PATH))
    N_files = CountFiles(SAVE_PATH)
    print("All images gathered will be saved along with existing items in this folder")

# start capturing the images from the camera.
# here we define the limit number of images taken
cap = cv2.VideoCapture(0)
frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)
num_samples = 200

print("press 'b' to capture background image")
print("Hit Space to Capture Images : ")

start = False # changes if we want to capture images
count = 0
if N_files>0:
    count = N_files # follows the number of images that been saved

while True:
    ret, frame = cap.read()
    k = cv2.waitKey(10)

    if not ret:
        continue

    if count == num_samples+N_files:
        break

    font = cv2.FONT_HERSHEY_SIMPLEX

    # save the background image (the reference to the subtraction)
    if k == ord('b'):
        img_background = frame

    if start:
        # image subtraction and converting to binary image
        img_sub = background_sub(img_background, frame)
        img_new = cv2.resize(img_sub, (227, 227), interpolation=cv2.INTER_AREA)
        img_new = np.where(img_new > 128, 255, 0)

        print(img_sub.shape)
        print(img_new.shape)

        # saving the image
        save_path = os.path.join(SAVE_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, img_new)
        count += 1

    frame_copy = frame
    cv2.putText(frame_copy, "Collecting {}".format(count),
            (5, 25), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame_copy)

    if k == ord(' '):
        start = not start

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, SAVE_PATH))
cap.release()
cv2.destroyAllWindows()


