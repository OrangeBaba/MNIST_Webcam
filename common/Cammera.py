import pyaudio
import wave
import numpy as np
import time
#from error_hider import noalsaerr
import matplotlib.pyplot as plt
import requests
import cv2 as cv
import datetime
import configparser

capture_device_index = 1

image_name = f'mnist_data\\Number.jpg'

def setup():  
    # camera setting
    config_read()
    cap = cv.VideoCapture(capture_device_index)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

def config_read():
    cfg = configparser.ConfigParser()
    cfg.read('./config.ini', encoding='utf-8')
    global capture_device_index
    capture_device_index = int(cfg['Device']['capture_device_index'])

def capture(image_name, frame):
    cv.imwrite(image_name, frame)
