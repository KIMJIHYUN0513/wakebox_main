import cv2
import dlib
from math import hypot

#relevant CO2
import RPi.GPIO as GPIO
import time

#relevant sound
import pygame


DO = 26
CO2 = 19
co2_num = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(DO, GPIO.IN)

def Sound():
    pygame.mixer.init()
    pygame.mixer.music.load("doorbell.wav")
    pygame.mixer.music.play(1)

def CO2():
    try:
        while True:
            if GPIO.input(DO) > 1 :
                print("%d  CO2 concentration < 1000ppm", GPIO.input.value)
            else :
                print("CO2 concentration >= 1000ppm")
                Sound()
            time.sleep(5)
            
    except KeyboardInterrupt:
        GPIO.cleanup()


# create default face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX
#mouth_points = [48, 49, 50, 51, 52, 53, 54, 55,
#                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
r_eye_points = [42, 43, 44, 45, 46, 47]
l_eye_poits = [36, 37, 38, 39, 40, 41]

count = 0
count_E = 0

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(image, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(image, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

capture = cv2.VideoCapture(1)

#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)480 320 160
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)360 240 120
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True :
    _, image = capture.read()

    # convert frame to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = detector(gray)
    if(len(faces) == 0):
        print("no face recog")
        count = count +1
    if (count%10 == 9): Sound()

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio(
            l_eye_poits, landmarks)
        right_eye_ratio = get_blinking_ratio(
            r_eye_points, landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blinking_ratio >= 6.0:
            print("blinking")
            count_E = count_E +1
        if (count_E%4 == 3): Sound()

    # show the frame
    cv2.imshow("Frame", image)
    if GPIO.input(DO) <= 1:
        co2_num = co2_num + 1

    if co2_num%5 == 4:
        print("CO2 concentration < 1500ppm")
        
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


