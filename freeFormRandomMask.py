from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from resizeimage import resizeimage
from PIL import Image
import cv2

def random_free_form(config):
	imageHeight, imageWidth = 256, 256
	maxVertex = 12
	maxLength = 100
	maxAngle = 2*np.pi/5
	meanAngle = 2*np.pi/15
	maxBrushWidth = 40

	mask = np.zeros([imageHeight,imageWidth], dtype='int')
	numVertex = int(random.uniform(4,maxVertex))
	startX = random.uniform(0,imageWidth)
	startY = random.uniform(0,imageHeight)
	brushwid = []

	X = []
	X.append(startX)
	Y = []
	Y.append(startY)

	fig, ax = plt.subplots()
	ax.axis('off')
	brushWidth = random.uniform(12,maxBrushWidth)
	
	for i in range(numVertex):
		angle = random.uniform(meanAngle - maxAngle,meanAngle + maxAngle)
		if(i%2 == 0):
			angle = 2*np.pi - angle
		length = random.uniform(0,maxLength)
		startX = startX + length*np.sin(angle)
		startY = startY + length*np.cos(angle)
		X.append(startX)
		Y.append(startY)
		cir = plt.Circle((startX,startY),brushWidth/4,color="black")
		ax.add_artist(cir)
	plt.plot(X,Y,linewidth=brushWidth,color="black")
	fig.savefig('mask64.png')

	with open('mask64.png', 'r+b') as f:
	    with Image.open(f) as image:
	        cover = resizeimage.resize_cover(image, [256, 256])
	        cover.save('mask256.png', image.format)
	im = cv2.imread("mask256.png",0)
	
	for x in range(256):
		for y in range(256):
			if(im[x][y] == 255):
				im[x][y]=1
			else:
				im[x][y]=0
	#img = Image.fromarray(im,'L')
	#img.save("mask1.png")
        
	im = np.expand_dims(im,axis = 0)
	im = np.expand_dims(im,axis = 3)
	return im
