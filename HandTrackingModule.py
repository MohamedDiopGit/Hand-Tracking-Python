import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Hand detection module

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,1,
                                        self.detectionCon, self.trackCon)  # voir paramètre par défaut, modifiale en fonction du souhait : static_image en false pour rapidité
        self.mpDraw = mp.solutions.drawing_utils  # dessine la main

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converti en RGB l'image pour hands (accepte slmt RGB image)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks) # vérifie la présence d'une main sur la cam (None ou coordonnées)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # pour chaque main / landMarks = position
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                                 self.mpHands.HAND_CONNECTIONS)  # pas le imgRGB car display sur img, handconnections pour les liaison
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):  # id = point de la main
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)  # 10 : largeur du cercle, B G R


        return lmList
def main():
    # FPS

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # 1 pour autre webcam
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0 :
            print(lmList[4])

        #FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()