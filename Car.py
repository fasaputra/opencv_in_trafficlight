from random import randint
import time
import imutils

class MyCar:
    tracks = []
    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
    def getRGB(self):
        return (self.R,self.G,self.B)
    def getTracks(self):
        return self.tracks
    def getId(self):
        return self.i
    def getState(self):
        return self.state
    def getDir(self):
        return self.dir
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x,self.y])
        self.x = xn
        self.y = yn
    def setDone(self):
        self.done = True
    def timedOut(self):
        return self.done
    def going_UP(self,mid_start,mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end: #cruzo la linea
                    state = '1'
                    self.dir = 'up'
                    return True
            else:
                return False
        else:
            return False
    def going_DOWN(self,mid_start,mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start: #cruzo la linea
                    state = '1'
                    self.dir = 'down'
                    return True
            else:
                return False
        else:
            return False
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True
class MultiCar:
    def __init__(self, cars, xi, yi):
        self.cars = cars
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        
def sliding_window(image, stepSize, windowSize):
    	# slide a window across the image
    	for y in range(0, image.shape[0], stepSize):
    		for x in range(0, image.shape[1], stepSize):
    			# yield the current window
    			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
     
    # keep looping over the pyramid
    while True:
    	# compute the new dimensions of the image and resize it
    	w = int(image.shape[1] / scale)
    	image = imutils.resize(image, width=w)
     
    	# if the resized image does not meet the supplied minimum
    	# size, then stop constructing the pyramid
    	if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
    		break
     
    	# yield the next image in the pyramid
    	yield image                