# Rock_Paper_Scissors
Building a computer program that plays Rock-Paper-Scissors, and **always** wins.
The working principle is simple - the computer recognize the hand gesture of the player, and do the move needed to win.
The recognition part should be in real-time that it will look like they are playing in the same time.

## Move Recognition
There are only four moves that can be done - rock, paper, scissors or none of them.
The recgnition of the player move is based on the computer's camera, using two methods:
* counting fingers, using basic OpenCV functions. 
* train a neural network to the classification between the moves.
In this project we will implement the two methods.

## Counting Fingers
### Step 1:
* First we need to detect the hand and mark her as contour.
* To do that we do image (or background) subtuction - basically we take 'snapshot' of an image without the hand in it, and subtract this image form any other image.
* After that we do some filtering to the image and turn are to be binary image.
### Step 2:
* Mark the largest contour in the image.
* Comment: after background subtraction and filtering we suppose that the largest contour we will get is the hand. 
### Step 3:
* Use Convex Hull method and find defects between the hull and the contour. This defects are the spaces between the fingers, and thats how we can count fingers.
### Step 4:
From here it's pretty straight forward to assume the player move - 
* 0 defects - rock.
* 1 defects - scissors.
* 2 or above defects - paper.
Comment: you can notice that this method we can't recognize the fourth option ("None"). For that we will use neural network that was trained to recgnize those situations. 

## Neural Network
### Step 1 - collect data:
* There is existing database of images of hands in rock-paper-scissors moves at differnt scin colors and hand shapes, generated by CGI.
* Here is a link to download this database - https://public.roboflow.com/classification/rock-paper-scissors
* Another way is to generate the data by myself. You can gather images, labeling and saving them using the script [gather_data.py](./gather_data.py). Here I choose to do also background subtraction, but it's not necessary.
### Step 2 - training:
* In this project I use existing nueral network named **MobileNet**, build for fast image classification.
* We changing only the last layers and do transfer learning for the new data.
* As already been said, we train the data by 4 catagories - rock, paper, scissors and None.
**Important Note - The MobileNet input image size is defined, therefore we need to resize the images and make sure there are 3 channels.**
### Step 3:
Now we have saved a model that predict for each given image what is the player move.
All we need to do is to react to this move accordingly.
