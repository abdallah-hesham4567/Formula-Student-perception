import cv2
import numpy as np
import random
## This function selects a random frame the video and gives you the ability to select a small group of pixels within to get hsv values


video_path = "data/Formula_test.mp4"     ## make sure to edit this to your video relative path


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(number_of_frames)
frame_number = np.random.randint(0,number_of_frames - 1)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

cap.release()   

if not ret:
    raise RuntimeError("Could not read video")


frame = cv2.resize(frame, (800,450))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


x, y, w, h = cv2.selectROI("Select Cone Region", frame, False)


roi_hsv = hsv[y:y+h, x:x+w]


pixels = roi_hsv.reshape(-1, 3)

mean_hsv = np.mean(pixels, axis=0)
min_hsv = np.min(pixels, axis=0)
max_hsv = np.max(pixels, axis=0)
median_hsv = np.median(pixels, axis=0)

print("\nHSV values from selected region")
print("--------------------------------")
print("Mean   :", mean_hsv)
print("Median :", median_hsv)
print("Minimum:", min_hsv)
print("Maximum:", max_hsv)

# Print some individual pixels too
print("\nFirst 20 HSV pixels:")
for pixel in pixels[:20]:
    print(pixel)

cv2.destroyAllWindows()