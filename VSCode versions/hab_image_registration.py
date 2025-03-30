import cv2
import numpy as np

# Kernel size for Gaussian blur (denoising)
BLUR_KERNEL_SIZE = (3, 3)

# Factor to downsample the images (0.5 reduces the resolution to half)
DOWNSAMPLE_FACTOR = 0.3

# Lowe's ratio test threshold
LOWES_RATIO_THRESHOLD = 0.1

# Load the images
referenceImage = cv2.imread('/Users/vasteven/Projects/HAB/2010 atkinson pool.png')
#capturedImage = cv2.imread('/Users/vasteven/Projects/HAB/2023 atkinson pool.png')
capturedImage = cv2.imread('/Users/vasteven/Projects/HAB/2023 atkinson rotated.png')


# Apply Gaussian Blur for de-noising
capturedImageBlurred = cv2.GaussianBlur(capturedImage, BLUR_KERNEL_SIZE, 0)
referenceImageBlurred = cv2.GaussianBlur(referenceImage, BLUR_KERNEL_SIZE, 0)

# Downsample images to reduce resolution and emphasize significant features
capturedImageBlurredResized = cv2.resize(capturedImageBlurred, (0,0), fx=DOWNSAMPLE_FACTOR, fy=DOWNSAMPLE_FACTOR)
referenceImageBlurredResized = cv2.resize(referenceImageBlurred, (0,0), fx=DOWNSAMPLE_FACTOR, fy=DOWNSAMPLE_FACTOR)

# Convert images to grayscale
capGray = cv2.cvtColor(capturedImageBlurredResized, cv2.COLOR_BGR2GRAY)
refGray = cv2.cvtColor(referenceImageBlurredResized, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
capKeypoints, capDescriptors = sift.detectAndCompute(capGray, None)
refKeypoints, refDescriptors = sift.detectAndCompute(refGray, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Perform the matching
matches = flann.knnMatch(capDescriptors, refDescriptors, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# Ratio test as per Lowe's paper
for i, (m,n) in enumerate(matches):
    if m.distance < LOWES_RATIO_THRESHOLD * n.distance:
        matchesMask[i] = [1,0]

# Convert matches to format required by drawMatchesKnn
good_matches = [m for m, mask in zip(matches, matchesMask) if mask[0]]

# Draw parameters
draw_params = dict(matchColor=(0,255,0),
                   singlePointColor=(255,0,0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)

# Drawing the matches
matched_image = cv2.drawMatchesKnn(capturedImageBlurredResized, capKeypoints, referenceImageBlurredResized, refKeypoints, matches, None, **draw_params)

# Display
cv2.imshow('Matches', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Note: For further steps, such as finding a homography and warping the image, you would proceed from the list of good matches.
