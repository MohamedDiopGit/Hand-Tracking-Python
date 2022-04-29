import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # 1 pour autre webcam

#Hand detection module

mpHands = mp.solutions.hands
hands = mpHands.Hands() #voir paramètre par défaut, modifiale en fonction du souhait : static_image en false pour rapidité
mpDraw = mp.solutions.drawing_utils  # dessine la main

# FPS
pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #converti en RGB l'image pour hands (accepte slmt RGB image)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks) # vérifie la présence d'une main sur la cam (None ou coordonnées)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:    #pour chaque main / landMarks = position
            for id, lm in enumerate(handLms.landmark): #id = point de la main
                #print(id,lm)
                h,w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)

                if id == 4:  # 0 : bas, 4 : pouce
                    cv2.circle(img,(cx,cy), 10 , (255,0,255), cv2.FILLED)  # 10 : largeur du cercle

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # pas le imgRGB car display sur img, handconnections pour les liaison

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

