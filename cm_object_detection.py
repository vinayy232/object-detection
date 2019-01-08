import cv2
import numpy as np


def main():

    lowerBound = np.array([33, 80, 40])
    upperBound = np.array([102, 255, 255])

    cam = cv2.VideoCapture(0)
    kernelOpen = np.ones((6, 6))
    kernelClose = np.ones((20, 20))

    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cam.read()
        img = cv2.resize(img, (340, 220))

        # convert BGR to HSV
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # create the Mask
        mask = cv2.inRange(imgHSV, lowerBound, upperBound)
        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        eroded = cv2.erode(dilated, np.ones((3, 3), np.uint8), iterations=1)
        mask = eroded
        # morphology
        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        maskFinal = maskClose
        _, conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(img, conts, -1, (255, 0, 0), 3)
        for i in range(len(conts)):
            x, y, w, h = cv2.boundingRect(conts[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, str(i + 1), (x, y + h), font, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            # cv2.cv.PutText(cv2.cv.fromarray(img), str(i + 1), (x, y + h), font, (0, 255, 255))
        cv2.imshow("maskClose", maskClose)
        cv2.imshow("maskOpen", maskOpen)
        cv2.imshow("mask", mask)
        cv2.imshow("cam", img)

        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    cam.release()


if __name__ == "__main__":
    main()