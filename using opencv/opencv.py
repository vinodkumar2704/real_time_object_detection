import cv2
import numpy as np

url='http://192.168.1.33:8080/video'

def nothing(x):
 pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 43, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 55, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 19, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 86, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0) #use url in case

while True:

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        # print(area)

        if area > 400:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            cx = int(M["m10"] / area)
            print(cx)
            # print("forward")
            if area > 20000:
                # for i in range(10000):
                print("stop")

            elif cx > 600:
                print("right")

            elif cx < 300:
                print("left")
            elif 300 < cx < 600:
                print("forward")

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)
    if key == 27:
         break

cap.release()
cv2.destroyAllWindows()