import cv2
import numpy as np
from sklearn.metrics import pairwise

# Declaring global variables that we will use throughout the function

background = None

accumulated_weight = 0.5

# Manually setting up our Region Of Interest (ROI) for grabbing the hand.

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


#Finding the average background value

"""The following function is used to calculate the weighted sum 
of input image src and the accumulator dst so that dst becomes a 
running average of a frame sequence"""

def calc_accum_avg(frame, accumulated_weight):
    '''
    Given a frame and a previous accumulated weight, computed
    the weighted average of the image passed in.
    '''
    
    # Grab the background
    global background
    
    # For first time, create the background from a copy of the frame.
    if background is None:
        background = frame.copy().astype("float")
        return None

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment(frame, threshold=25):
    global background
    
    # Calculates the Absolute Difference between the background and the passed in frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Apply a threshold to the image so we can grab the foreground
    # We only need the threshold, so we will throw away the first item in the tuple with an underscore _
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours from the image
    # Again, only grabbing what we need here and throwing away the rest
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list is 0, then we didn't grab any contours!
    if len(contours) == 0:
        return None
    else:
        # Given the way we are using the program, the largest external contour should be the hand (largest by area)
        # This will be our segment
        hand_segment = max(contours, key=cv2.contourArea)
        
        # Return both the hand segment and the thresholded hand image
        return (thresholded, hand_segment)


def count_fingers(thresholded, hand_segment):
    # Calculate the convex hull of the hand segment
    conv_hull = cv2.convexHull(hand_segment)
    
    # Find the top, bottom, left, and right points of the convex hull
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # Calculate the center of the hand
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # Calculate the maximum Euclidean distance between the center and the extreme points
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()

    # Create a circular ROI with a radius 80% of the maximum distance
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    # Create a circular ROI mask
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

    # Apply the circular ROI mask to the thresholded image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Find contours in the circular ROI
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Initialize finger count
    count = 0

    # Loop through the contours to count the fingers
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Check if the contour is not part of the wrist and if the contour is small enough to be a finger
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        limit_points = ((circumference * 0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1

    return count

cam = cv2.VideoCapture(0)

# Intialize a frame count
num_frames = 0

# keep looping, until interrupted
while True:
    # get the current frame
    ret, frame = cam.read()

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    frame_copy = frame.copy()

    # Grab the ROI from the frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # Apply grayscale and blur to ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # For the first 30 frames we will calculate the average of the background.
    # We will tell the user while this is happening
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",frame_copy)
            
    else:
        # now that we have the background, we can segment the hand.
        
        # segment the hand region
        hand = segment(gray)

        # First check if we were able to actually detect a hand
        if hand is not None:
            
            # unpack
            thresholded, hand_segment = hand

            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

            # Count the fingers
            fingers = count_fingers(thresholded, hand_segment)

            # Display count
            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Also display the thresholded image
            cv2.imshow("Thesholded", thresholded)

    # Draw ROI Rectangle on frame copy
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)

    # increment the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.imshow("Finger Count", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()