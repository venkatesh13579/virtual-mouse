import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import autopy
import math

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.tipIds = [4, 8, 12, 16, 20]
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = int(detectionCon * 100)  # Convert to int
        self.trackCon = int(trackCon * 100)  # Convert to int

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon / 100.0,  # Convert back to float
            min_tracking_confidence=self.trackCon / 100.0  # Convert back to float
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 0), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def show_instructions():
    """Displays the instructions for using the virtual mouse."""
    instructions = [
        "Virtual Mouse Controls:",
        "",
        "1. Check for scroll up gesture (all fingers up).",
        "2. Check for scroll down gesture (all fingers up).",
        "3. Check for right arrow gesture.",
        "4. Check for left arrow gesture",
        "5. Volume up gesture (Index and middle fingers up)",
        "6. Volume down gesture (Index and middle fingers up)",
        "7. Index, middle, and ring fingers up: Right-clicking Mode",
        "Both Index and middle fingers are up: Clicking Mode.",
        "",
        "Press 's' to start using the Virtual Mouse.",
    ]
    img = np.zeros((580, 740, 3), dtype=np.uint8)
    y = 50
    for line in instructions:
        cv2.putText(img, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
    return img


def main():
    ##########################
    wCam, hCam = 640, 480
    frameR = 100  # Frame Reduction
    smoothening = 5
    #########################

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        instructions_img = show_instructions()
        cv2.imshow("Instructions", instructions_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to start
            break
    cv2.destroyWindow("Instructions")

    detector = handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    while True:
        #  Find hand Landmarks
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)

        if len(lmlist) != 0:
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)

            # Only index finger up: Moving mode
            if fingers[1] == 1 and fingers[2] == 0:
                # 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                # Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                #Move Mouse
                autopy.mouse.move(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Both Index and middle fingers are up: Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineinfo = detector.findDistance(8, 12, img)
                if length < 30:
                    cv2.circle(img, (lineinfo[4], lineinfo[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

            # Index, middle, and ring fingers up: Right-clicking Mode
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                length, img, lineinfo = detector.findDistance(8, 12, img)
                if length < 30:
                    cv2.circle(img, (lineinfo[4], lineinfo[5]),
                               15, (0, 255, 255), cv2.FILLED)
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)

            # Check for scroll up gesture (all fingers up)
            if fingers == [0, 1, 1, 1, 1]:
                pyautogui.scroll(60)

            # Check for scroll down gesture (thumb and pinky up)
            if fingers == [1, 0, 0, 0, 1]:
                pyautogui.scroll(-60)

            # Check for right arrow gesture
            if fingers == [1, 0, 0, 0, 0]:
                pyautogui.press('right')

            # Check for left arrow gesture
            if fingers == [0, 0, 0, 0, 1]:
                pyautogui.press('left')

                # Volume up gesture (Index and middle fingers up)
            if fingers == [0, 1, 1, 0, 1]:
                pyautogui.press('volumeup')

                # Volume down gesture (Thumb and pinky fingers up)
            if fingers == [0, 1, 1, 0, 0]:
                pyautogui.press('volumedown')


            # Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # 12. Display
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



