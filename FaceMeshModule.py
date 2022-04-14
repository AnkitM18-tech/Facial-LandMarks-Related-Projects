import cv2
import mediapipe as mp
import time
import math

class FaceMeshDetector():
    def __init__(self,staticMode = False,max_num_faces = 2,refine_L = True,min_detection_conf = 0.5,min_tracking_conf = 0.5):
        self.staticMode = staticMode
        self.max_num_faces = max_num_faces
        self.refine_L = refine_L
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.max_num_faces,self.refine_L,self.min_detection_conf,self.min_tracking_conf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    def findFaceMesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih,iw,ic = img.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)
                    cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,.5,(0,255,0),1)
                    # print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        return img,faces

    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success,img = cap.read()
        img, faces = detector.findFaceMesh(img)
        # if len(faces)!=0:
        #     print(len(faces))
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f"FPS: {int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow("Image",img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

if __name__ == '__main__':
    main()