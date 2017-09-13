#coding:utf-8
import cv2
import numpy as np
from math import *
import os
import time

#const
W = 32
H = 19
ii = np.zeros(714,444)

def calcSum(f,x,y,H,W):
	return f[x+H, y+W] - f[x, y+W] - f[x+H, y] + f[x, y]


def findCharacter(newx, newy, characters, already, diff_const):
	minDiff = diff_const
	Cha = (0,0) 
	for (xx,yy) in characters:
		if not((xx,yy) in already):
			temp = sqrt((xx-newx)*(xx-newx)+(yy-newy)*(yy-newy))
			if temp<=minDiff:
				minDiff = temp
				Cha = (xx,yy)

	if (minDiff == diff_const):
		return 0
	else:
		return Cha


def check(x,y,H,W,characters):
	ans = []
	diff_const = sqrt(W*W + H*H) 
	#Assume (x,y) is 姓
	Cha_xing = (x,y)
	ans.append(Cha_xing)
	#Find 名
	Cha_ming = findCharacter(x, y+2*W, characters, ans, diff_const)
	if (Cha_ming == 0):
		return []
	else:
		ans.append(Cha_ming)
	
	#Find 性
	Cha_xing1 = findCharacter(x+2.5*H, y, characters, ans, diff_const)
	if (Cha_xing1 == 0):
		return []
	else:
		ans.append(Cha_xing1)

	#Find 别
	Cha_bie = findCharacter(Cha_xing1[0], Cha_xing1[1] + 2 * W, characters, ans, diff_const)
	if (Cha_bie == 0):
		return []
	else:
		ans.append(Cha_bie)
	
	#Find 出
	Cha_chu = findCharacter(Cha_xing1[0] + 2.5*H, Cha_xing1[1], characters, ans, diff_const)
	if (Cha_chu == 0):
		return []
	else:
		ans.append(Cha_chu)

	#Find 生
	Cha_sheng = findCharacter(Cha_chu[0], Cha_chu[1] + 2*W, characters, ans, diff_const)
	if (Cha_sheng == 0):
		return []
	else:
		ans.append(Cha_sheng)

	#Find 地
	Cha_di = findCharacter(Cha_chu[0] + 2.5*H, Cha_chu[1], characters, ans, diff_const)
	if (Cha_di == 0):
		return []
	else:
		ans.append(Cha_di)

	#Find 址
	Cha_zhi = findCharacter(Cha_di[0], Cha_di[1] + 2*W, characters, ans, diff_const)
	if (Cha_zhi == 0):
		return []
	else:
		ans.append(Cha_zhi)

	#Find 民
	Cha_min = findCharacter(Cha_bie[0], Cha_bie[1] + 7*W, characters, ans, diff_const)
	if (Cha_min == 0):
		return []
	else:
		ans.append(Cha_min)

	#Find 族
	Cha_zu = findCharacter(Cha_min[0], Cha_min[1] + 2*W, characters, ans, diff_const)
	if (Cha_zu == 0):
		return []
	else:
		ans.append(Cha_zu)

	#Find 年
	Cha_nian = findCharacter(Cha_min[0] + 2.5*H, Cha_min[1], characters, ans, diff_const)
	if (Cha_nian == 0):
		return []
	else:
		ans.append(Cha_nian)

	#Find 月
	Cha_yue = findCharacter(Cha_nian[0], Cha_nian[1] + 4*W, characters, ans, diff_const)
	if (Cha_yue == 0):
		return []
	else:
		ans.append(Cha_yue)

	#Find 日
	Cha_ri = findCharacter(Cha_yue[0], Cha_yue[1] + 4*W, characters, ans, diff_const)
	if (Cha_ri == 0):
		return []
	else:
		ans.append(Cha_ri)
	
	return ans


def RotateImg(img, degree):
	height,width=img.shape[:2]
	#旋转后的尺寸
	heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
	widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))	

	matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)	

	matRotation[0,2] +=(widthNew-width)/2  #重点在这步，目前不懂为什么加这步
	matRotation[1,2] +=(heightNew-height)/2  #重点在这步	

	return cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))	


def findMatching(ii, size, threshold):
	target = []
	use = []

	start = time.time()

	for x in range(size[0]-H):
		for y in range(size[1]-W):
			rec1 = ii[x+H, y+W] - ii[x, y+W] - ii[x+H, y] + ii[x, y]
			rec2 = ii[x+H, y+W*3/4] - ii[x, y+W*3/4] - ii[x+H, y+W/4] + ii[x, y+W/4]
			feature = rec1 - 2*rec2
			if (feature > threshold):
				target.append([x,y,feature])
				use.append(0)

	end = time.time()
	print("2:", end - start)

	# target = np.array(target)
	# target = target[target[:,2].argsort()]
	
	target.sort(key=lambda rec : rec[2],reverse = True)
	cleanTarget = []
	
	start = time.time()

	for i in range(len(target)):
		if (use[i] == 0):
			count = 1
			use[i] = 1
			for j in range(i+1,len(target)):
				if (use[j] == 0):
					if (abs(target[i][0]-target[j][0]) <= H)and(abs(target[i][1]-target[j][1]) <= W/2):
						use[j] = 1
			cleanTarget.append((int(target[i][0]),int(target[i][1])))	
	
	# for (x,y) in cleanTarget:
	# 	cv2.rectangle(img, (y+int(W/4),x), (y+int(W*3/4), x+H), (0,255,0), 1)
	
	end = time.time()
	print("3:", end - start)

	# cleanTarget = np.array(cleanTarget)
	# cleanTarget = cleanTarget[cleanTarget[:,0].argsort()]
	cleanTarget.sort(key = lambda rec: rec[1])

	start = time.time()

	# for i in xrange(cleanTarget.shape[0]):
	# 	x = cleanTarget[i][0]
	# 	y = cleanTarget[i][1]
	# 	if (y <= (size[1]/4)):
	# 		ans = check(x,y,H,W/2,cleanTarget)
	# 		if not(ans == []):
	# 			return ans

	for (x,y) in cleanTarget:
		if (y <= (size[1]/4)):
			ans = check(x,y,H,W/2,cleanTarget)
			if (ans != []):
				end = time.time()
	 			print("4:", end - start)
				return ans
	return []


def IDcardMatching(img):
	#preprocessing
	
	height, width  = img.shape[:2]
	if (height > width):
		img = RotateImg(img, -90)
	img = cv2.resize(img, (714,444), interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
	gray = cv2.GaussianBlur(gray, (5,5), 1.5)
	mean, std = cv2.meanStdDev(gray)
	gray = (gray-mean) / std * 45 + 128;

	
	#calc matrix
	size = gray.shape
	s = np.zeros(size)
	ii = np.zeros(size)

	start = time.time()

	for x in range(size[0]):
		for y in range(size[1]):
			if (y == 0):
	 			s[x,y] = gray[x,y]
 			else:
	 			s[x,y] = s[x,y-1] + gray[x,y]
			if (x == 0):
				ii[x,y] = s[x,y]
			else:
				ii[x,y] = ii[x-1,y] + s[x,y] 

	end = time.time()
	print("1:", end - start)

	threshold = 4000
	result = findMatching(ii, size, threshold)
	# while (result == []):
	# 	threshold = threshold - 500
	# 	if (threshold == 3000):
	# 		break
	# 	result = findMatching(ii, size, threshold)
	
	if (result == []):
		return 0, img

	for (x,y) in result:
		cv2.rectangle(img, (y+int(W/4),x), (y+int(W*3/4), x+H), (0,0,255), 1)

	return 1, img

if __name__ == '__main__':
	rootdir = "/Users/Taylor/Lianlian/idcard/arrangedIdcard"
	l = os.listdir(rootdir)
	for i in range(0, 10):
		path = os.path.join(rootdir, l[i])
		img = cv2.imread(path)
		result, img_result = IDcardMatching(img)
		if result  == 1:
			cv2.imwrite("./idcard/success_idcard/"+str(i)+".jpg", img_result)
		else:
			cv2.imwrite("./idcard/fail_idcard/"+str(i)+".jpg", img_result)

