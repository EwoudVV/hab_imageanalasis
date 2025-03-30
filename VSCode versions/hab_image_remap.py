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
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Perform the matching
matches = flann.knnMatch(capDescriptors, refDescriptors, k=2)

# Apply Lowe's ratio test
good_matches = [m for m, n in matches if m.distance < LOWES_RATIO_THRESHOLD * n.distance]

# Extract location of good matches
points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

for i, match in enumerate(good_matches):
    points1[i, :] = capKeypoints[match.queryIdx].pt
    points2[i, :] = refKeypoints[match.trainIdx].pt

# Compute the affine transformation matrix
M, inliers = cv2.estimateAffinePartial2D(points1, points2)

# Apply the affine transformation
transformed_image = cv2.warpAffine(capturedImage, M, (referenceImage.shape[1], referenceImage.shape[0]))

# Show the original and transformed image
cv2.imshow('Original', capturedImage)
cv2.imshow('Transformed', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
