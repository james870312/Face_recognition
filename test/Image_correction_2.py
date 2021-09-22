import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#images = glob.glob('../image_files/Calibration_data/*.png')

img = cv.imread('../image_files/Calibration_data/left13.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (7,6), None)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # Draw and display the corners
    cv.drawChessboardCorners(img, (7,6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey()
        
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#img = cv.imread('../image_files/Calibration_data/left13.png')
h, w = img.shape [:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h) )

'''
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult', dst)
cv.waitKey()
#cv.imwrite('calibresult.png', dst)
'''

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult', dst)
cv.waitKey()
#cv.imwrite('calibresult.png', dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )





stereoMatcher = cv.StereoBM_create()
left = cv.imread('../image_files/Calibration_data/left13.png')
right = cv.imread('../image_files/Calibration_data/right13.png')

fixedLeft = cv.remap(left, mapx, mapy, interpolation=cv.INTER_LINEAR)
fixedRight = cv.remap(right, mapx, mapy, interpolation=cv.INTER_LINEAR)

grayLeft = cv.cvtColor(fixedLeft, cv.COLOR_BGR2GRAY)
grayRight = cv.cvtColor(fixedRight, cv.COLOR_BGR2GRAY)
depth = stereoMatcher.compute(grayLeft, grayRight)

DEPTH_VISUALIZATION_SCALE = 2048
cv.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)

#cv.imshow('remap', depth)
cv.waitKey()



        
        

cv.destroyAllWindows()
