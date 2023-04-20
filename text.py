import cv2
import numpy as np
import os

# Generate positive samples
cmd = "/opt/homebrew/opt/opencv@3/bin/opencv_createsamples -img /Users/biggles/Documents/Code/Blocks/pos2/img22n1.png -bg negative.txt -info info.txt -num 90 -w 7 -h 7 -vec positive_samples.vec"
os.system(cmd)

# Train classifier
cmd = "/opt/homebrew/opt/opencv@3/bin/opencv_traincascade -data classifier -vec positive_samples.vec -bg negative.txt -numPos 90 -numNeg 10 -numStages 10 -w 7 -h 7"
os.system(cmd)

# Test classifier
cascade = cv2.CascadeClassifier('classifier/cascade.xml')
img = cv2.imread('img.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blocks = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), maxSize=(80, 80))
