import socket
import car_dir
import motor
import time
import re
import pyttsx3

corpus = [
    "we just reached out first attractions, we will begin tour after 10 seconds",
    "we just reached out second attractions, we will begin tour after 10 seconds"
    ]

motor_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
motor_socket.connect(('192.168.25.9',1325))

car_dir.setup(busnum=1)
motor.setup(busnum=1)
car_dir.home()

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate',rate-30)

def tts_guide(text):
    engine.say(text)    # location corresponding text
    engine.runAndWait()

while True:
    data = ''
    data = motor_socket.recv(1024)
        
    if not data:
        break
    else:
        print(data)
    
    if '1' in re.findall('1',data): #find voice1 command in burst data
        print('tts command received')
        motor.stop()
        print('tts speaching')
        tts_guide(corpus[0])
    elif '2' in re.findall('2',data): #find voice2 command in burst data
        print('tts command received')
        motor.stop()
        print('tts speaching')
        tts_guide(corpus[1])
    
    if data == "Forward-Right":
        motor.forwardWithSpeed(70)
        car_dir.turn_right()
    elif data == "Forward-Left":
        motor.forwardWithSpeed(70)
        car_dir.turn_left()
        motor.forward()
    elif data == "Forward":
        car_dir.home()
        motor.forwardWithSpeed(70)
    elif data == "Home":
        car_dir.home()
        motor.stop()
    elif data == "Stop" or "Stop" in re.findall("Stop",data):
        motor.stop()

motor_socket.close()
