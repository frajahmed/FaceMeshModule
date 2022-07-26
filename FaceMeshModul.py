import cv2
import mediapipe as mp
import time
import math


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=1, refLandmarks=False, minDetectionCon=0.5, minTrackCon=0.5):

        self.refLandmarks = refLandmarks

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refLandmarks, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findFaceMesh(self, image, draw=True):
        self.imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    #self.mpDraw.draw_landmarks(image, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                    #self.mpDraw.draw_landmarks(image, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec,
                                               #self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    #self.mpDraw.draw_landmarks(image, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                    #self.mpDraw.draw_landmarks(image, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,
                                               #self.mp_drawing_styles.get_default_face_mesh_contours_style())

###########oder nur das#############
                    self.mpDraw.draw_landmarks(image, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
###################################


                face = []
                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = image.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                     #           0.7, (0, 255, 0), 1)

                    #print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        return image, faces

    def findDistance(self,p1, p2, image=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if image is not None:
            cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, image
        else:
            return length, info


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=1)
    while True:
        success, image = cap.read()
        image, faces = detector.findFaceMesh(image)
        if len(faces)!= 0:
            print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()