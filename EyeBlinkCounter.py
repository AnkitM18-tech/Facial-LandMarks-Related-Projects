import cv2
import cvzone
from cvzone.FaceMeshModule import  FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Capturing the Frames
cap = cv2.VideoCapture("./Video.mp4")
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640,360,[20,50],invert=True)

# Eyes points
idList = [22,23,24,26,110,157,158,159,160,161,130,243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255,0,255)

while True:
    # When the video reaches the total frame count we reset it to 0
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    # detecting the face mesh
    success,img = cap.read()
    img,faces = detector.findFaceMesh(img,draw=False)
    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img,face[id],5,color,cv2.FILLED)
        # Finding distance b/w top and bottom and Left and right edge of the eye
        leftUp = face[159]
        leftDown = face[23]
        leftLeftEdge = face[130]
        leftRightEdge = face[243]
        verLength,_ = detector.findDistance(leftUp,leftDown)
        horLength,_ = detector.findDistance(leftLeftEdge,leftRightEdge)
        cv2.line(img,leftUp,leftDown,(0,200,0),3)
        cv2.line(img,leftLeftEdge,leftRightEdge,(0,200,0),3)
        # print(horLength)
        # Finding the ratio between distance of the eye sides
        ratio = int((verLength/horLength)*100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)
        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            color = (0,200,0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255,0,255)
        # Putting the text inside the image
        cvzone.putTextRect(img,f"Blink Count : {blinkCounter}",(50,100),colorR=color)
        imgPlot = plotY.update(ratioAvg,color)
        # cv2.imshow("Video Image Plot",imgPlot)
        img = cv2.resize(img,(640,360))
        imgStack = cvzone.stackImages([img,imgPlot],2,1)
    else:
        img = cv2.resize(img,(640,360))
        imgStack = cvzone.stackImages([img,imgPlot],2,1)

    cv2.imshow("Video Captured",imgStack)
    key = cv2.waitKey(25)
    if key == ord("q"):
        break