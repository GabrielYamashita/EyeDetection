''' Pupil Detection '''
#--------------------------------
# Date : 23-06-2020
# Project : Pupil Detection
# Category : Detection
# Company : weblineindia
# Department : AI/ML
#--------------------------------
import os
import numpy as np
import cv2
from skimage import io
import imutils
import dlib


def pixelReader(img,startHorizontal,startVertical,height):
	'''
	Used to read specific pixel of given image.
	img = image 
	startHorizontal =  horizontal starting pixel value
	startVertical = vertical starting value
	height = maximum limit for pixel reading
	'''
	# list for satisfied pixels
	blackColour = []

	# for loops for traversing pixels
	for j in range(-int(height*1.5), 0):
		for i in range(startVertical- height,startVertical+int(height*1.5)):
			# setting lower bound for pixels, termination 
			blackLowerRange = [80, 50, 50]
			pixel = startHorizontal + j
			colorCI = img[int(pixel), i]
			if ((colorCI[0] <= blackLowerRange[0] and colorCI[1] <= blackLowerRange[1] and colorCI[2] <= blackLowerRange[2])):
				blackColour.append([int(pixel), i])
	return blackColour

def getFaceAttributeVector(image):
	'''
	Used to get the 68 facial attributes of face image.
	image: it is an image.
	'''
	# dlib shape predictor 
	predictorPath = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictorPath)
	dets = detector(image)

	# if dlib is able to detect the faces 
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	for k, d in enumerate(dets):
		shape = predictor(image, d)

	# to get the 68 facial points detect from dlib into a list
	faceCoord = np.empty([68, 2], dtype = int)

	for b in range(68):
		faceCoord[b][0] = shape.part(b).x
		faceCoord[b][1] = shape.part(b).y

	return faceCoord

def getEyeCoordinates(image, faceCoord):
	'''
	Used to crop the eyes from image
	image: it is an image.
	faceCoord: array of facial landmarks coordinates
	'''
	# using facial points to crop the part of the eyes from image
	leftEye = image[int(faceCoord[19][1]):int(faceCoord[42][1]),int(faceCoord[36][0]):int(faceCoord[39][0])]
	rightEye = image[int(faceCoord[19][1]):int(faceCoord[42][1]),int(faceCoord[42][0]):int(faceCoord[45][0])]

	eyeLCoordinate = [int(faceCoord[37][0]+int((faceCoord[38][0]-faceCoord[37][0])/2)), int(faceCoord[38][1]+int((faceCoord[40][1]-faceCoord[38][1])/2))]
	eyeRCoordinate = [int(faceCoord[43][0]+int((faceCoord[44][0]-faceCoord[43][0])/2)), int(faceCoord[43][1]+int((faceCoord[47][1]-faceCoord[43][1])/2))]

	leftBlackPixel = pixelReader(image,eyeLCoordinate[1],eyeLCoordinate[0],int((faceCoord[38][0]-faceCoord[37][0])/2))
	rightBlackPixel = pixelReader(image, eyeRCoordinate[1], eyeRCoordinate[0], int((faceCoord[44][0]-faceCoord[43][0])/2))
	return leftEye, rightEye, leftBlackPixel, rightBlackPixel

def getPupilPoint(img, blackCoordinates, eyeTopPointX, eyeBottomPointY):
	'''
	Used to get the coordinates of the pupil 
	image: cropped eye image
	blackCoordinates: It is the array of the black color pixels inside the eyes
	eyeTopPointX: eye's starting coordinates horizontally
	eyeBottomPointY: eye's starting coordinates vertically
	'''

	# after getting the eyes pixels applying cv2 method to detect circle pixels which we can do using houghcircles 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT, dp = 1,minDist = 5,
		param1=250,param2=10,minRadius=1,maxRadius=-1)
	
	# check if houghcircles has detect any circle or not
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:1]:
			pupilPoint = [int(eyeTopPointX[0])+ i[0],int(eyeBottomPointY[1]) + i[1]]
	else:
		# if HoughCircles is unable to detect the circle than using eyes points to get pupil points 
		a = 0
		for j,k in blackCoordinates:
			if a == int(len(blackCoordinates)/2):
				pupilPoint = [k,j]
			else:
				raise Exception("F de novo")
			a += 1
	return pupilPoint

# Reading the image 
cap = cv2.VideoCapture(0)

while True:
	has_frame, frame = cap.read()
	
	# Stop the program if reached end of video
	if not has_frame:
		print('Camera not working')
		break


	faceVector = getFaceAttributeVector(frame)
	leftEye, rightEye, eyeLeftBlackPixels, eyeRightBlackPixels = getEyeCoordinates(frame, faceVector)

	leftEyeCoord, eyeBrowCoord = faceVector[36], faceVector[19]
	# leftPupilPoint = getPupilPoint(leftEye, eyeLeftBlackPixels, leftEyeCoord, eyeBrowCoord)

	# getting pupilpoint for right eye
	# rightEye is the cropped part of face image


	rightEyeCoord = faceVector[42]
	print(leftEye)
	cv2.rectangle(frame,(leftEyeCoord[0],leftEyeCoord[1]-20),(leftEyeCoord[0]+40,leftEyeCoord[1]+20),(0,255,0),1)
	cv2.rectangle(frame,(rightEyeCoord[0],rightEyeCoord[1]-20),(rightEyeCoord[0]+40,rightEyeCoord[1]+20),(0,255,0),1)

	# rightPupilPoint = getPupilPoint(rightEye, eyeRightBlackPixels, rightEyeCoord, eyeBrowCoord)

	# drawing pupil points on image
	# cv2.circle(frame, tuple(leftPupilPoint), 5, (255, 0, 0), -1)
	# cv2.circle(frame, tuple(rightPupilPoint), 5, (255, 0, 0), -1)

	finalImage = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	cv2.imshow("AAAAAAAAAAAA", finalImage)

	key = cv2.waitKey(1)
	if key == ('c'):
		break

cap.release()
cv2.destroyAllWindows()

