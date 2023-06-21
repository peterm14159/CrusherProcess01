import imutils
from scipy.spatial import distance as dist
from imutils import perspective
import cv2
import numpy as np
import csv





def eval_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ms = np.mean(gray)
    return ms

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



def process(orig, c):
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(
        box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    for (x1, y1) in box:
        cv2.circle(orig, (int(x1), int(y1)), 1, (0, 0, 255), -1)

   	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 1, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 1, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 1, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 1, (255, 0, 0), -1)

	# draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            (255, 0, 255), 1)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            (255, 0, 255), 1)

	# compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
    ##if pixelsPerMetric is None:
    ##    pixelsPerMetric = goodTooth / 1

	# compute the size of the object
    ##dimA = dA / pixelsPerMetric
    ##dimB = dB / pixelsPerMetric
    dimA = dA
    dimB = dB

    dimA = max(dimA,dimB)


	# draw the object sizes on the image
    cv2.putText(orig, "{:.0f}".format(dimA),
                (int(tltrX - 15), int(tltrY + 30)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)
    return orig, dimA


def removeOutliers(values):
    # make sure max. is 100 & nan is 0
    values[values>100] = 100
    values[np.isnan(values)] = 0
    # remove zeros a[a != 0]
    # can rely on percentile between two values (eg. 25&75) or mean+ 1 std or both
    a = np.mean(values[values != 0])
    b = np.std(values[values != 0])
    #c = np.percentile(values[values != 0],50)
    #d = np.percentile(values[values != 0],75)
    #e = np.percentile(values[values != 0],25)
    result = []
    for y in values:
        if y >= (a-b) and y <= (a+b):
            result.append(y)
    if len(result) == 0:
        result = 0
    return result


def removeOutliersPMG(measurements):
#provides a single value representing the likely average of a set of measurements
#no valid measurements are returned with 1.0

    cnt = 0.0
    Mavg0 = np.mean(measurements[measurements != 1.0])
    Mstd0 = np.std(measurements[measurements != 1.0])
    numMeas = len(measurements)
    measFlag = np.ones(numMeas,dtype='int')
    mm = np.empty(measurements.shape, dtype = 'float')


    #for meas in measurements:
    #    if np.abs(meas-Mavg) < 1.0*Mstd and meas > 0.0 and meas != 1.0:
    #        mm[int(cnt)] = meas
    #        cnt += 1.0
    #    else:
    #        mm[int(cnt)] = 1.0

    for i in range(0,numMeas):
        if measurements[i] == 1.0:
            measFlag = 0

    Mrange = 10 #how close to be grouped
    numValid = measFlag.sum()
    validMeas = np.zeros(numValid,dtype='float')
    idex = 0
    for i in range(0,numMeas):
        if measFlag[i] == 1.0:
            validMeas[idex] = measurements[i]
            idex += 1

    clusters = np.zeros([numValid,numValid],dtype='int')
    clusterSum = np.zeros(numValid,dtype='int')
    for i in range(0,numValid):
        for j in range(0,numValid):
            if np.abs(validMeas[i] - validMeas[j] ) < Mrange:
                clusters[i,j] = 1
        for j in range(0,numValid):
            clusterSum[i] += clusters[i,j]

    Mmax = 0
    maxDex = 0
    for i in range(0,numValid):
        if clusterSum[i] > Mmax:
            Mmax = clusterSum[i]
            maxDex = i

    finalMeas = np.zeros(Mmax)
    dex = 0
    for j  in range(0,numValid):
        if clusters[maxDex,j] == 1:
            finalMeas[dex] = validMeas[j]
            dex += 1


    refinedMeasurement = np.median(finalMeas)
    conf = float(Mmax)/float(numMeas) * (1.0 - finalMeas.std()/finalMeas.mean())
    if conf < 0:
        conf = 0.0

    #conf = float(Mmax)/float(numMeas)
    #if cnt > 0.0:
    #    refinedMeasurement = np.median(mm[mm != 1.0])
    #    #conf = 1.0 - 2.0*float(np.std(mm[mm != 1.0])/np.mean(mm[mm != 1.0]))
    #    conf = cnt/numMeas*(1.0 - float(np.std(mm[mm != 1.0]))/refinedMeasurement)
    #    conf = float(cnt)/float(numMeas)

    print('M= ', measurements)
    print('f= ', measFlag)
    print(Mmax,refinedMeasurement, conf )
    #input("Press Enter to continue...")


    return refinedMeasurement , conf


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def click_eventL(event, x, y, flags, params):
   global cflag
   global idex
   if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')

        # put coordinates as text on the image
        #cv2.putText(drumImage, f'({x},{y})',(x,y),
        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for i in range(0,boxNum):
            if x > bboxMatL[i,0] and x < bboxMatL[i,2] and y > bboxMatL[i,1] and y < bboxMatL[i,3]:
                #boxImagesLm[i] = boxImagesL[i]
                drumLm[bboxMatL[i,1]:bboxMatL[i,3],bboxMatL[i,0]:bboxMatL[i,2],:] = boxImagesL[i]
                cflag = 1
                idex = i


def click_eventR(event, x, y, flags, params):
   global cflag
   global idex
   if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')

        # put coordinates as text on the image
        #cv2.putText(drumImage, f'({x},{y})',(x,y),
        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for i in range(0,boxNum):
            if x > bboxMatR[i,0] and x < bboxMatR[i,2] and y > bboxMatR[i,1] and y < bboxMatR[i,3]:
                #boxImagesLm[i] = boxImagesL[i]
                drumRm[bboxMatR[i,1]:bboxMatR[i,3],bboxMatR[i,0]:bboxMatR[i,2],:] = boxImagesR[i]
                cflag = 1
                idex = i



def  ReviewToothMeasurements(drumImage,drumImageM,measurements,direction):

    print('Review Measurements for drum moing in ', direction, ' direction')
    cv2.imshow('xxx', drumImage)
    cv2.moveWindow('xxx', 0,0)
    cv2.waitKey()

    cv2.imshow('xxx', drumImageM)
    cv2.moveWindow('xxx', 0,0)
    cv2.waitKey()

    NewMeasurements = measurements

    return NewMeasurements
