#Review Measurements
#display measurements image for each drum, right click callback
# leads to new user measurement ot replace automated one
#written by C-CORE: Peter McGuire Feb 2023

# Main Code
import matplotlib.pyplot as plt
import cv2
import numpy as np
#from vidstab.VidStab import VidStab
#from scipy.signal import find_peaks
from imutils import contours
import csv
import easygui
import os


def click_eventL(event, x, y, flags, params):
   global cflag
   global idex
   #global boxNum
   if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')

        # put coordinates as text on the image
        #cv2.putText(drumImage, f'({x},{y})',(x,y),
        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for i in range(0,boxNum):
           #print(bboxMatL[i,0],bboxMatL[i,1],bboxMatL[i,2],bboxMatL[i,3])
           if x > bboxMatL[i,0] and x < bboxMatL[i,2] and y > bboxMatL[i,1] and y < bboxMatL[i,3]:
                #boxImagesLm[i] = boxImagesL[i]
                drumLm[bboxMatL[i,1]:bboxMatL[i,3],bboxMatL[i,0]:bboxMatL[i,2],:] = boxImagesL[i]
                cflag = 1
                idex = i
                print('CFLAG = ',cflag)



def click_eventR(event, x, y, flags, params):
   global cflag
   global idex
   #global boxNum
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



#main
def review():
    global boxNum
    global cflag

    dispFlag = 'true'
    drumLimage = 'DrumScanL.jpg'
    drumLimageMeas = 'DrumScanL_meas.jpg'
    drumRimage = 'DrumScanR.jpg'
    drumRimageMeas = 'DrumScanR_meas.jpg'
    bboxFileNameL = 'leftDrumFrame.npy'
    bboxFileNameR = 'rightDrumFrame.npy'
    fileMeasL = 'drumL_measurements.npy'
    fileMeasR = 'drumR_measurements.npy'
    fileMeasL_new = 'drumL_measurements_new'
    fileMeasR_new = 'drumR_measurements_new'
    teethPerColumn = 12 # teeth per column
    columnsPerDrum = 12 #number of columns per drum

    boxNum = teethPerColumn*columnsPerDrum #number of inspection zones per drum
    global boxImagesL
    global boxImagesR
    boxImagesL = np.empty([boxNum],dtype=object)
    boxImagesR = np.empty([boxNum],dtype=object)
    boxImagesLm = np.empty([boxNum],dtype=object)
    boxImagesRm = np.empty([boxNum],dtype=object)
    global bboxMatR
    global bboxMatL

    bboxMatR = (np.floor(np.load(bboxFileNameR))).astype(int)
    bboxMatL = (np.floor(np.load(bboxFileNameL))).astype(int)
    drumL = cv2.imread(drumLimage)
    drumR = cv2.imread(drumRimage)
    global drumLm
    global drumRm
    drumLm = cv2.imread(drumLimageMeas)
    drumRm = cv2.imread(drumRimageMeas)
    measurementsL = np.load(fileMeasL)
    measurementsR = np.load(fileMeasR)


    #extract  boxes
    shiftXL = 0 #shift variable for user input
    shiftYL = 0
    shiftXR = 0 #shift variable for user input
    shiftYR = 0
    dispNameR = 'Right Drum'
    dispNameL = 'Left Drum'
    for cnt in range (0, boxNum):
        bboxMatR[cnt,0] += shiftYR
        bboxMatR[cnt,1] += shiftXR
        bboxMatR[cnt,2] += shiftYR
        bboxMatR[cnt,3] += shiftXR
        bboxMatL[cnt,0] += shiftYL
        bboxMatL[cnt,1] += shiftXL
        bboxMatL[cnt,2] += shiftYL
        bboxMatL[cnt,3] += shiftXL


        if bboxMatR[cnt,0] < 0:
            bboxMatR[cnt,0] = 0
        if bboxMatR[cnt,1] < 0:
            bboxMatR[cnt,1] = 0
        if bboxMatR[cnt,2] >= drumR.shape[1]:
            bboxMatR[cnt,2] = drumR.shape[1]-1
        if bboxMatR[cnt,3] >= drumR.shape[0]:
            bboxMatR[cnt,3] = drumR.shape[0]-1

        if bboxMatL[cnt,0] < 0:
            bboxMatL[cnt,0] = 0
        if bboxMatL[cnt,1] < 0:
            bboxMatL[cnt,1] = 0
        if bboxMatL[cnt,2] >= drumL.shape[1]:
            bboxMatL[cnt,2] = drumL.shape[1]-1
        if bboxMatL[cnt,3] >= drumL.shape[0]:
            bboxMatL[cnt,3] = drumL.shape[0]-1

        #cropL = np.array(drumL[bboxMatL[cnt,1]:bboxMatL[cnt,3],bboxMatL[cnt,0]:bboxMatL[cnt,2],:])
        #cropR = np.array(drumR[bboxMatR[cnt,1]:bboxMatR[cnt,3],bboxMatR[cnt,0]:bboxMatR[cnt,2],:])
        #boxImagesL[cnt] = cropL
        #boxImagesR[cnt] = cropR
        boxImagesL[cnt] = drumL[bboxMatL[cnt,1]:bboxMatL[cnt,3],bboxMatL[cnt,0]:bboxMatL[cnt,2],:]
        boxImagesR[cnt] = drumR[bboxMatR[cnt,1]:bboxMatR[cnt,3],bboxMatR[cnt,0]:bboxMatR[cnt,2],:]
        boxImagesLm[cnt] = drumLm[bboxMatL[cnt,1]:bboxMatL[cnt,3],bboxMatL[cnt,0]:bboxMatL[cnt,2],:]
        boxImagesRm[cnt] = drumRm[bboxMatR[cnt,1]:bboxMatR[cnt,3],bboxMatR[cnt,0]:bboxMatR[cnt,2],:]

        if dispFlag == 'true':
            cv2.imshow(dispNameR, boxImagesLm[cnt])
            cv2.moveWindow(dispNameR, 100,100)
            cv2.imshow(dispNameL, boxImagesRm[cnt] )
            cv2.moveWindow(dispNameL, 300,100)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

    #Display marked up drum scan image and
    #Left Drum
    cv2.namedWindow(dispNameL)
    cv2.setMouseCallback(dispNameL, click_eventL)
    cflag = 0
    global idex
    idex = 9999
    while True:
        #print(cflag)
        cv2.imshow(dispNameL,drumLm)
        cv2.moveWindow(dispNameL, 0,0)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
          break
        #check if click is inside a box
        if cflag == 1:
            print('idex =', idex)
            cflag = 0
            bbox = cv2.selectROI(dispNameL, drumLm)
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(drumLm, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(drumLm, "{:.0f}".format(w),
                    (int(x + w/2), int(y + h/2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
            print(x,y,w,h)
            row = idex%teethPerColumn
            col = int(idex/columnsPerDrum)
            measurementsL[row,col] = w
            cv2.setMouseCallback(dispNameL, click_eventL)


    print(measurementsL)
    fileMeasL_new = None
    while fileMeasL_new is None:
        fileMeasL_new= easygui.filesavebox(title='Specify Left Drum Measurement .csv file',filetypes=['*.csv'])


    fileMeasL_new1= os.path.splitext(fileMeasL_new)
    np.save(fileMeasL_new1[0]+'.npy',measurementsL)
    f = open(fileMeasL_new1[0] + '.csv' , 'w')
    print('writing: ', fileMeasL_new  )
    writer = csv.writer(f)
    writer.writerows(np.floor(measurementsL))
    f.close()
    cv2.imwrite('DrumScanL_meas_new.jpg', drumLm)
    cv2.destroyAllWindows()


    #Right Drum
    cv2.namedWindow(dispNameR)
    cv2.setMouseCallback(dispNameR, click_eventR)
    cflag = 0
    idex = 9999
    while True:
        cv2.imshow(dispNameR,drumRm)
        cv2.moveWindow(dispNameR, 0,0)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
          break
        #check if click is inside a box
        if cflag == 1:
            print('idex =', idex)
            cflag = 0
            bbox = cv2.selectROI(dispNameR, drumRm)
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(drumRm, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(drumRm, "{:.0f}".format(w),
                (int(x + w/2), int(y + h/2)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)
            print(x,y,w,h)
            row = idex%teethPerColumn
            col = int(idex/columnsPerDrum)
            measurementsR[row,col] = w
            cv2.setMouseCallback(dispNameR, click_eventR)


    print(measurementsR)
    fileMeasR_new = None
    while fileMeasR_new is None:
        fileMeasR_new = easygui.filesavebox(title='Specify Right Drum Measurement .csv file',filetypes=['*.csv'])
    fileMeasR_new1= os.path.splitext(fileMeasL_new)

    np.save(fileMeasR_new1[0]+'.npy',measurementsR)
    f = open(fileMeasR_new1[0] + '.csv' , 'w')
    print('writing: ', fileMeasR_new  )
    writer = csv.writer(f)
    writer.writerows(np.floor(measurementsR))
    f.close()
    cv2.imwrite('DrumScanR_meas_new.jpg', drumRm)
