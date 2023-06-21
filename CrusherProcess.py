import matplotlib.pyplot as plt
import cv2
import numpy as np
#from vidstab.VidStab import VidStab
#from scipy.signal import find_peaks
from imutils import contours
from Utilities import *
import csv
import easygui
import sys
import ReviewMeasurements



def click_event(event, x, y, flags, params):
   global mouseX,mouseY
   if event == cv2.EVENT_LBUTTONDOWN:
      #print(f'({x},{y})')
      mouseX,mouseY = x,y
      drumImageCopy[:,:,:] = drumImage[:,:,:]
      # put coordinates as text on the image
      cv2.putText(drumImageCopy, f'({x},{y})',(x,y),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

      # draw point on the image
      cv2.circle(drumImageCopy, (x,y), 3, (0,255,255), -1)
      #xskip = int(drumImageCopy.shape[1] / 13.)
      #yskip = int(drumImageCopy.shape[0] / 13.)
      #wid = int(drumImageCopy.shape[1] / 13.)
      #hid = int(drumImageCopy.shape[0] / 13.)

      for i in range(0,12):
          for j in range (0,12):
              xnew = x+ i*xskip + (j%2)*int(xskip/2)
              ynew = y + j*yskip
             # if i%2 == 0:
             #   cv2.rectangle(drumImageCopy, (xnew, ynew), (xnew + wid, ynew + hid), (255, 0, 0), 2)
             # else:
             #   cv2.rectangle(drumImageCopy, (xnew, ynew), (xnew + wid, ynew + hid), (255, 0, 255), 2)




def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def click_eventR(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      drumImageCopy[:,:,:] = drumImage[:,:,:]
      # put coordinates as text on the image
      cv2.putText(drumImageCopy, f'({x},{y})',(x,y),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

      # draw point on the image
      cv2.circle(drumImageCopy, (x,y), 3, (0,255,255), -1)
      #xskip = 104
      #yskip = 72
      #wid = 100
      #hid = 60
      for i in range(0,12):
          for j in range (0,12):
              xnew = x+ i*xskip + (j%2)*int(xskip/2)
              ynew = y + j*yskip
              #if i%2 == 0:
              #  cv2.rectangle(drumImageCopy, (xnew, ynew), (xnew + wid, ynew + hid), (255, 255, 0), 2)
              #else:
              #  cv2.rectangle(drumImageCopy, (xnew, ynew), (xnew + wid, ynew + hid), (255, 0, 255), 2)

def CreateLineScanPatt(frames, direction, bbox):
    #fileIn -> array of sequential video frames
    #dir -> direction of object movement in video (left/right)
    #pixLoc -> sub-window for extraction (x,y,w,h)
    #frRange -> (firstFrame,lastFrame)

    print('function: CreateLineScanPatt')
    print('Creating scan image from ', len(frames), ' frames.')
    print('Object rotation is : ', direction)

    (x, y, w, h) = [int(v) for v in bbox]
    frameNo = len(frames)
    dimX = frames[0].shape[1]
    dimY = frames[0].shape[0]
    dimZ =  frames[0].shape[2]
    #print(frameNo,dimX,dimY,dimZ)
    #print(w,h)

    scanImage = np.ones([h+int(0.25*h), w*frameNo,3], dtype='uint8')*127
    #frame = np.zeros([dimX,dimY,dimZ], dtype='uint8')

    dispName = 'making scan image'
    cv2.namedWindow(dispName, cv2.WINDOW_GUI_NORMAL)
    #cv2.namedWindow(dispName, cv2.WINDOW_KEEPRATIO)

    trim = min(8,int(h*0.05))
    ymin = scanImage.shape[0]
    ymax = 0
    for i in range(0,frameNo-1):
        frame = frames[i]
        if direction == 'right':
            #print(x,w)
            #cropped = np.array(frame[y+trim:y+h-trim,x-w:x,:],dtype='uint8')
            cropped = np.array(frame[y+trim:y+h-trim,x:x+w,:],dtype='uint8')
            if i == 0:
                locx = frameNo*w - w - 1
                locy = int((scanImage.shape[0]-cropped.shape[0])/2)
            else:
                #loc = frameNo*w-i*w-1
                loc = locx  #set to last found location
                scanImageGray = cv2.cvtColor(scanImage[:,loc-w:loc+w], cv2.COLOR_BGR2GRAY)
                croppedGray = cv2.cvtColor(cropped[:,int(w/2):w], cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(scanImageGray,croppedGray,cv2.TM_SQDIFF_NORMED)
                #cv2.imshow('xxx',scanImage)
                #cv2.imshow('yyy',res/res.max()*255)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                locx = min_loc[0] + loc-w-int(w/2)
                locy = min_loc[1]
                #print('loc= ',res.shape,min_loc,max_loc,min_val,max_val,locx,locy)
            #print(x,w,locx,h,locy,cropped.shape,scanImage.shape)
            scanImage[locy:locy+h-2*trim,(locx):locx+w] = cropped
        elif direction == 'left':
            cropped = np.array(frame[y+trim:y+h-trim,x:x+w,:],dtype='uint8')
            if i == 0:
                locx = 0
                locy = int((scanImage.shape[0]-cropped.shape[0])/2)
            else:
                #loc = frameNo*w-i*w-1
                loc = locx  #set to last found location
                scanImageGray = cv2.cvtColor(scanImage[:,loc:loc+w], cv2.COLOR_BGR2GRAY)
                croppedGray = cv2.cvtColor(cropped[:,0:int(w/2)], cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(scanImageGray,croppedGray,cv2.TM_SQDIFF_NORMED)
                #cv2.imshow('xxx',scanImage)
                #cv2.imshow('yyy',res/res.max()*255)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                locx = min_loc[0] + loc
                locy = min_loc[1]
                #print('loc= ',res.shape,min_loc,max_loc,min_val,max_val,locx,locy)
            scanImage[locy:locy+h-2*trim,(locx):locx+w] = cropped

        if locy > ymax:
            ymax = locy
        elif locy < ymin:
            ymin = locy
        cv2.imshow(dispName, cropped)
        cv2.resizeWindow(dispName,w*1,h*1)
        cv2.moveWindow(dispName, 0,0)
        cv2.waitKey(1)


    #Finalize image
    if direction == 'right':
        xmin = locx
        xmax = scanImage.shape[1]
    if direction == 'left':
        xmin = 0
        xmax = locx+w
    scanImageTrim = scanImage[ymin-1:ymax+h-2*trim+1,xmin:xmax,:]
    #cv2.imshow(dispName, scanImageTrim)
    #cv2.resizeWindow(dispName,scanImageTrim.shape[1],scanImageTrim.shape[0])
    #cv2.moveWindow(dispName, 0,0)
    #cv2.waitKey()
    cv2.destroyAllWindows()
    return scanImageTrim.copy()

def CreateLineScan(frames, direction, bbox):
    #fileIn -> array of sequential video frames
    #dir -> direction of object movement in video (left/right)
    #pixLoc -> sub-window for extraction (x,y,w,h)
    #frRange -> (firstFrame,lastFrame)

    print('Creating scan image from ', len(frames), ' frames.')
    print('Object rotation is : ', direction)

    (x, y, w, h) = [int(v) for v in bbox]
    frameNo = len(frames)
    dimX = frames[0].shape[1]
    dimY = frames[0].shape[0]
    dimZ =  frames[0].shape[2]
    print(frameNo,dimX,dimY,dimZ)
    print(w,h)

    scanImage = np.zeros([h, w*frameNo,3], dtype='uint8')
    #frame = np.zeros([dimX,dimY,dimZ], dtype='uint8')

    dispName = 'making scan image'
    cv2.namedWindow(dispName, cv2.WINDOW_GUI_NORMAL)
    #cv2.namedWindow(dispName, cv2.WINDOW_KEEPRATIO)


    for i in range(0,frameNo-1):
        frame = frames[i]
        if direction == 'right':
            cropped = np.array(frame[y:y+h,x-w:x,:])
            loc = frameNo*w-i*w-1
            scanImage[:,(loc-w):loc] = cropped
        elif direction == 'left':
            cropped = np.array(frame[y:y+h,x:x+w,:])
            loc = 0 + i*w
            scanImage[:,loc:(loc+w)] = cropped


        cv2.imshow(dispName, cropped)
        cv2.resizeWindow(dispName,w*1,h*1)
        cv2.moveWindow(dispName, 0,0)
        cv2.waitKey(1)


        #cropped_image = np.array(frames[y:(y+h),x:(x+w)])
    cv2.imshow(dispName, scanImage)
    cv2.resizeWindow(dispName,scanImage.shape[1],scanImage.shape[0])
    cv2.moveWindow(dispName, 0,0)
    cv2.waitKey()



#main
#ReviewMeasurements.review()

#Select Video File
#fileIn = r'D:\Crusher\Videos\DC11\33B2F0AE-1070-4074-AB81-865F999F509D.mov'
#wobbly
#fileIn = r'D:\Crusher\Videos\DC11\66416BC0-5424-4B8A-A2EE-A478C0B99235.mov'
fileIn = easygui.fileopenbox('Select Video File')
if fileIn is None: # User closed msgbox
    sys.exit(0)

#load video file
cap = cv2.VideoCapture(fileIn)
dimX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
dimY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(np.round(cap.get(cv2.CAP_PROP_FPS), 0))
drumRPS = 0.93  #drum rotation speed
first_frame = 0
for choice in ['Left', 'Right']:

    if choice == 'Left':
        direction = 'right'
    else:
        direction = 'left'


    #user input for 1st frame
    reply = 0
    while reply == 0:
        dispName = 'Initiate Scan'
        cv2.namedWindow(dispName, cv2.WINDOW_GUI_NORMAL)
        #cv2.namedWindow(dispName, cv2.WINDOW_AUTOSIZE)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        for fnum  in range(first_frame, n_frames-first_frame-1):
            check, frame = cap.read()
            if check == True:
                frameDisp = frame.copy()
                if fnum == first_frame:
                    print('Specify the extraction region.')
                    print('Use the mouse to select a thin vertical slice of the drum moving in ', direction, ' direction.')
                    print('Use the space key to advance video frames.')
                    print('Press esc when ready.')
                    cv2.imshow(dispName, frameDisp)
                    cv2.moveWindow(dispName, 0,0)
                    bbox = cv2.selectROI(dispName, frameDisp)
                    (x, y, w, h) = [int(v) for v in bbox]
                    w = 12
                    bboxMod = np.array([x,y,w,h])


                cv2.rectangle(frameDisp, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.imshow(dispName, frameDisp)
                cv2.moveWindow(dispName, 0,0)
                k = cv2.waitKey()
                if k == 27:
                    first_frame = fnum
                    break
        cv2.destroyAllWindows()


        #extract enough frames for one drum rotation
        dispName = 'Scanning...'
        cv2.namedWindow(dispName, cv2.WINDOW_GUI_NORMAL)

        last_frame = first_frame+int(fps*1.0/drumRPS+fps*2.0/(drumRPS*12))  # for image stitching
        #last_frame = first_frame+int(fps*1.0/drumRPS+fps*1.0/(drumRPS*12)) + int(fps*1.0/drumRPS)   # for image stitching
        expNoTeeth = 14
        scan_lines = last_frame-first_frame  # for image stitching
        print(' ')
        print('Utilizing frames ', first_frame,' through ', last_frame)
        frames = np.empty([scan_lines],dtype=object)
        cnt = 0
        for fnum  in range(first_frame, last_frame-1):
            check, frame = cap.read()
            if check == True:
                frames[cnt] = np.array(frame)
                if cnt == 0:
                    framesAvg = np.zeros(frame.shape, dtype='int')
                    framesAvg = cv2.add(framesAvg,frames[0].astype('int'))
                else:
                    framesAvg = cv2.add(framesAvg,frames[cnt].astype('int'))
                #cv2.imshow(dispName, frame)
                #cv2.moveWindow(dispName, 0,0)
                #cv2.waitKey(1)
                cnt += 1
                #ramesAvgtmp = cv2.normalize(framesAvg,None, 0, 255, cv2.NORM_MINMAX)
                #cv2.imshow('xxx', framesAvgtmp.astype('uint8'))
                #cv2.moveWindow('xxx', 0,0)
                #cv2.waitKey(1)

        cv2.destroyAllWindows()
        #framesAvg /= cnt
        cv2.normalize(framesAvg,framesAvg, 0, 255, cv2.NORM_MINMAX)

        #cv2.imshow('xxx', framesAvg.astype('uint8'))
        #cv2.moveWindow('xxx', 0,0)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        dispName = 'scanImage review'
        #cv2.namedWindow(dispName, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(dispName, cv2.WINDOW_FULLSCREEN)


        print('scanning movement in ' , direction, '  direction')
        drumImage = CreateLineScanPatt(frames, direction, bboxMod)
        cv2.imshow(dispName, drumImage)
        cv2.moveWindow(dispName, 0,0)
        #reply = easygui.boolbox("Keep this image?", "", ["Yes", "No"])
        #cv2.waitKey(1)
        cv2.destroyAllWindows()


    #Align analysis boxes
        easygui.msgbox("Align grid with teeth\n -> Click upper left of image to position grid\n "
                       "-> box height/width -> (w & s) / (a & d)\n -> box vertical/horizontal spacing -> (i & k) / (j & l) \n "
                       "-> rotate image o/p")
    #reply = 0
    #while reply == 0:
        key = 0
        dispName = 'Align Analysis Boxes'
        xskip = 112
        yskip = 76
        wid = 106
        hid = 74
        xskip = int(drumImage.shape[1] / 13.)
        xskip = int(drumImage.shape[1] / expNoTeeth)
        yskip = int(drumImage.shape[0] / 13.)
        wid = int(drumImage.shape[1] / 13.)
        wid = int(drumImage.shape[1] / expNoTeeth)
        hid = int(drumImage.shape[0] / 13.)

        cv2.namedWindow(dispName, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(dispName, drumImage)
        cv2.moveWindow(dispName, 0,0)
        cv2.setMouseCallback(dispName, click_event)
        drumImageCopy = drumImage.copy()
        print(' ')
        print('Line scan complete')
        print('Align grid with teeth')
        print('Click upper left of image to position grid')
        print('box height/width -> (w & s) / (a & d)')
        print('box vertical/horizontal spacing -> (i & k) / (j & l) ')


        while key != 27:
            cv2.imshow(dispName, drumImageCopy)
            cv2.moveWindow(dispName, 0,0)
            key = cv2.waitKey(1) & 0xFF
            #print(key)
            if key  ==  ord('o'): #97: # 61:  #a
                drumImage = rotate_image(drumImage,0.5)
                drumImageCopy = rotate_image(drumImageCopy,0.5)
                #cv2.imshow(dispName, drumImageCopy)
                #cv2.moveWindow(dispName, 0,0)
            elif key == ord('p'):
                drumImage = rotate_image(drumImage,-0.5)
                drumImageCopy = rotate_image(drumImageCopy,-0.5)
                #cv2.imshow(dispName, drumImageCopy)
                #cv2.moveWindow(dispName, 0,0)
            elif key == ord('a'):
                wid = wid+5
            elif key == ord('d'):
                wid = wid-5
            elif key == ord('w'):
                hid = hid+5
            elif key == ord('s'):
                hid = hid-5
            elif key == ord('j'):
                xskip += 1
            elif key == ord('l'):
                xskip -= 1
            elif key == ord('i'):
                yskip += 1
            elif key == ord('k'):
                yskip -= 1
            #print('x , y = ', mouseX,mouseY)

            try:
                mouseX
            except NameError:
                locx = 0
                locy = 0
            else:
                drumImageCopy[:,:,:] = drumImage[:,:,:]
                xloc = mouseX
                yloc = mouseY
                for i in range(0,12):
                    for j in range (0,12):
                        xnew = xloc+ i*xskip + (j%2)*int(xskip/2)
                        ynew = yloc + j*yskip
                        if i %2 == 0:
                            cv2.rectangle(drumImageCopy, (xnew, ynew), (xnew + wid, ynew + hid), (255, 255, 0), 2)
                        else:
                            cv2.rectangle(drumImageCopy, (xnew, ynew), (xnew + wid, ynew + hid), (255, 0, 255), 2)
        frameParams = [xloc,yloc,xskip,yskip,wid,hid]

        #xloc = mouseX
        #yloc = mouseY
        #click_event(cv2.EVENT_LBUTTONDOWN,mouseX,mouseY)
        #print('x , y = ', mouseX,mouseY)
        cv2.destroyAllWindows()


        gaussKernSize = (11,13)
        gaussKernSize = (21,21)
        toothSize = (70,25)
        teethPerColumn = 12 # teeth per column
        columnsPerDrum = 12 #number of columns per drum
        measurements = np.empty([teethPerColumn,columnsPerDrum],dtype='int')
        #set up tooth pattern for matching
        toothImg = np.zeros([200,200], dtype='uint8')
        toothImg = cv2.rectangle(toothImg, (10,10), (10+toothSize[0], 10+toothSize[1]), (255,255,255), -1)
        tooth_cnt = cv2.findContours(toothImg, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        tooth_cnt = tooth_cnt[0]

        #old style processing
        for i in range(0,12):
            for j in range (0,12):
                xnew = xloc + i*xskip + (j%2)*int(xskip/2)
                ynew = yloc + j*yskip
                drumSubImageGray = cv2.cvtColor(drumImage[ynew:ynew+hid,xnew:xnew+wid,:],cv2.COLOR_BGR2GRAY)
                gray = drumSubImageGray
                gray = cv2.GaussianBlur(gray, gaussKernSize, 0)
                cv2.normalize(gray,gray, 0, 255, cv2.NORM_MINMAX)
                edged = auto_canny(gray)
                edged = cv2.dilate(edged, None, iterations=2)
                edged = cv2.erode(edged, None, iterations=2)
                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                #if dispFlag == 'true':
                #    cv2.imshow(dispName, edged)
                #    cv2.moveWindow(dispName, 100,100)
                #    cv2.waitKey(1)

                tmp0  = edged.copy()
                cv2.drawContours(tmp0, cnts[0], -1, (255,0,0), 1)
                #if dispFlag == 'true':
                #    cv2.imshow(dispName, tmp0)
                #    cv2.moveWindow(dispName, 100,100)
                #    cv2.waitKey(1)


                cnts = cnts[0]
                if len(cnts) > 0:
                    # sort the contours from left-to-right and initialize the
                    (cnts, _) = contours.sort_contours(cnts)

                    #select contour
                    cA = np.empty(len(cnts))
                    cE = np.empty(len(cnts))
                    cR = np.empty(len(cnts))
                    cS = np.empty(len(cnts))
                    for k in range(0 , len(cnts)):
                        cA[k] = cv2.contourArea(cnts[k])
                        cS[k] = cv2.matchShapes(cnts[k],tooth_cnt[0],1,0.0)
                        x,y,w,h = cv2.boundingRect(cnts[k])
                        rect_area = w*h
                        cE[k] = cA[k]/rect_area #extent
                        cR[k] = w/h  #aspect ratio
                        if w < h :
                            cS[k] = 99999.0
                            print('w,h = ', w,h)
                        if cS[k] < 0.0001:
                            cS[k] = 99999.0
                            print('modified cS[k]')
                        if w >= wid - 1:
                            cS[k] = 99999.0
                        if cE[k] < 0.9:
                            cS[k] = 99999.0
                        if h > 35:
                             cS[k] = 99999.0
                        if w > 95:
                              cS[k] = 99999.0
                        cR[k] = cA[k]/cS[k]

                    indtouse = np.argmax(cA)   #this is the main contour selection criteria, could also use aspect ratio, extent, match shapes
                    indtouse = np.argmin(cS)
                    indtouse = np.argmax(cR)

                    c = cnts[indtouse].copy()
                    #orig = boxImages[j].copy()
                    orig = drumSubImageGray
                    origDisp = drumSubImageGray.copy()
                    cv2.drawContours(origDisp, cnts, indtouse, (0,255,0), 3)
                    #cv2.drawContours(origDisp, c, -1, (0,255,0), 3)

                    #if dispFlag == 'true':
                    #    cv2.imshow(dispName, origDisp)
                    #    cv2.moveWindow(dispName, 100,100)
                    #    cv2.waitKey(1)

                    # Use old code to extract measurements
                    orig, dimA = process(orig, c)
                    #if dispFlag == 'true':
                    #    cv2.imshow(dispName, orig)
                    #    cv2.moveWindow(dispName, 100,100)
                    #    cv2.waitKey(1)

                else:
                    #orig = boxImages[peakInd[i,j],j].copy()
                    #orig = boxImages[j].copy()
                    orig = drumSubImageGray
                    dimA = 0

                #record measurements
                print('dimA = ', dimA)
                row = j%teethPerColumn
                col = int(j/columnsPerDrum)
                measurements[row,col] = int(dimA)
                drumImageCopy[ynew:ynew+hid,xnew:xnew+wid,0] = orig
                drumImageCopy[ynew:ynew+hid,xnew:xnew+wid,1] = orig
                drumImageCopy[ynew:ynew+hid,xnew:xnew+wid,2] = orig


                cv2.imshow('xxx', orig)
                cv2.moveWindow('xxx', 0,0)
                cv2.waitKey(1)

        cv2.imshow('xxx', drumImageCopy)
        cv2.moveWindow('xxx', 0,0)
        #cv2.waitKey()
        reply = easygui.boolbox("Keep this result?", "", ["Yes", "No"])

    ##need to convert from from Params to bboxMat for review
    cnt = 0
    BoxNum = 12*12 #number of inspection zones
    bboxMat = np.empty([BoxNum,4])
    for i in range(0,12):
      for j in range (0,12):
          x = frameParams[0]
          y = frameParams[1]
          xskip = frameParams[2]
          yskip = frameParams[3]
          wid = frameParams[4]
          hid = frameParams[5]
          xnew = x+ i*xskip + (j%2)*int(xskip/2)
          ynew = y + j*yskip
          bboxMat[cnt,0] = xnew
          bboxMat[cnt,1] = ynew
          bboxMat[cnt,2] = xnew + wid
          bboxMat[cnt,3] = ynew + hid
          cnt += 1


    if choice == 'Left':
        drumImageL = drumImage
        drumImageLCopy = drumImageCopy
        measurementsL = measurements
        frameParamsL = frameParams
        bboxMatL = bboxMat
    elif choice == 'Right':
        drumImageR = drumImage
        drumImageRCopy = drumImageCopy
        measurementsR = measurements
        frameParamsR = frameParams
        bboxMatR = bboxMat

#Save Images/Data to files
cv2.imwrite('DrumScanL.jpg', drumImageL)
cv2.imwrite('DrumScanL_meas.jpg', drumImageLCopy)
cv2.imwrite('DrumScanR.jpg', drumImageR)
cv2.imwrite('DrumScanR_meas.jpg', drumImageRCopy)

np.save('drumL_measurements',measurementsL)
np.save('drumR_measurements',measurementsR)

np.save('leftDrumFrame',bboxMatL)
np.save('rightDrumFrame',bboxMatR)


ReviewMeasurements.review()

#Review Measurements
#ReviewToothMeasurements(drumImageL,drumImageLCopy,measurementsL,'right')
#ReviewToothMeasurements(drumImageR,drumImageRCopy,measurementsR,'right')

exit(0)





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#Pattern matching approach
#set tooth pattern
print('select tooth pattern')
dispName = 'find patterns'
bbox = cv2.selectROI(dispName, drumImage)
(xtmp, ytmp, wtmp1, htmp1) = [int(v) for v in bbox]
templateGray1 = cv2.cvtColor(np.array(drumImage[ytmp:ytmp+htmp1,xtmp:xtmp+wtmp1,:],dtype='uint8'),cv2.COLOR_BGR2GRAY)
#templateGray1 = np.zeros([25,65],dtype='uint8')
#templateGray1[2:23,2:63] = 255
#wtmp = 65
#htmp = 25
templateGray1 = cv2.normalize(templateGray1,None,0,255,cv2.NORM_MINMAX)
bbox = cv2.selectROI(dispName, drumImage)
(xtmp, ytmp, wtmp2, htmp2) = [int(v) for v in bbox]
templateGray2 = cv2.cvtColor(np.array(drumImage[ytmp:ytmp+htmp2,xtmp:xtmp+wtmp2,:],dtype='uint8'),cv2.COLOR_BGR2GRAY)
#templateGray2 = np.zeros([25,65],dtype='uint8')
#templateGray2[2:23,2:33] = 255
templateGray2 = cv2.normalize(templateGray2,None,0,255,cv2.NORM_MINMAX)
bbox = cv2.selectROI(dispName, drumImage)
(xtmp, ytmp, wtmp3, htmp3) = [int(v) for v in bbox]
templateGray3 = cv2.cvtColor(np.array(drumImage[ytmp:ytmp+htmp3,xtmp:xtmp+wtmp3,:],dtype='uint8'),cv2.COLOR_BGR2GRAY)
#templateGray3 = np.zeros([25,65],dtype='uint8')
templateGray3 = cv2.normalize(templateGray3,None,0,255,cv2.NORM_MINMAX)
#drumImageGray = cv2.cvtColor(drumImage,cv2.COLOR_BGR2GRAY)
print(wid,wtmp1,hid,htmp1,xloc, yloc)
print(templateGray1.shape,templateGray2.shape,templateGray3.shape)
print(drumImage.shape)
for i in range(0,12):
    for j in range (0,12):
        xnew = xloc + i*xskip + (j%2)*int(xskip/2)
        ynew = yloc + j*yskip
        drumSubImageGray = cv2.cvtColor(drumImage[ynew:ynew+hid,xnew:xnew+wid,:],cv2.COLOR_BGR2GRAY)
        drumSubImageGray = cv2.normalize(drumSubImageGray,None,0,255,cv2.NORM_MINMAX)
        print(drumSubImageGray.shape,ynew,xnew)
        res1 = cv2.matchTemplate(drumSubImageGray,templateGray1,cv2.TM_SQDIFF_NORMED)
        res2 = cv2.matchTemplate(drumSubImageGray,templateGray2,cv2.TM_SQDIFF_NORMED)
        res3 = cv2.matchTemplate(drumSubImageGray,templateGray3,cv2.TM_SQDIFF_NORMED)
        boxLoc1 = np.where(res1 == res1.min())
        boxLoc1[1][0] += xnew
        boxLoc1[0][0] += ynew
        boxLoc2 = np.where(res2 == res2.min())
        boxLoc2[1][0] += xnew
        boxLoc2[0][0] += ynew
        boxLoc3 = np.where(res3 == res3.min())
        boxLoc3[1][0] += xnew
        boxLoc3[0][0] += ynew

        if res1.min() <= res2.min() and res1.min() <= res3.min():
            cv2.rectangle(drumImageCopy, (boxLoc1[1][0], boxLoc1[0][0]), (boxLoc1[1][0] + wtmp1, boxLoc1[0][0] + htmp1), (0, 255, 0), 2)
        elif res2.min() <= res1.min() and res2.min() <= res3.min():
            cv2.rectangle(drumImageCopy, (boxLoc2[1][0], boxLoc2[0][0]), (boxLoc2[1][0] + wtmp2, boxLoc2[0][0] + htmp2), (255, 0, 0), 2)
        elif res3.min() <= res1.min() and res3.min() <= res2.min():
            cv2.rectangle(drumImageCopy, (boxLoc3[1][0], boxLoc3[0][0]), (boxLoc3[1][0] + wtmp3, boxLoc3[0][0] + htmp3), (0, 0, 255), 2)


cv2.namedWindow(dispName, cv2.WINDOW_GUI_NORMAL)
cv2.imshow(dispName, drumImageCopy)
cv2.moveWindow(dispName, 0,0)
cv2.waitKey()
