__author__ = 'zhengwang'

import threading
import socketserver
import socket
import serial
import cv2
import numpy as np
import math
import os
import sys
import select

# distance data measured by ultrasonic sensor
sensor_data = " "
server_ip = "192.168.25.9"

class NeuralNetwork(object):
    def __init__(self):
        self.model = None

    def create(self, layer_sizes):
        # create neural network
        self.model = cv2.ml.ANN_MLP_create()
        self.model.setLayerSizes(np.int32(layer_sizes))
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 0.01))

    def load_model(self, path):
        if not os.path.exists(path):
            print("Model 'nn_model.xml' does not exist, exit")
            sys.exit()
        self.model = cv2.ml.ANN_MLP_load(path)
        
    def predict(self, X):
        ret, resp = self.model.predict(X)
        return resp.argmax(-1)

class RCControl(object):
    def __init__(self):
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.bind((server_ip,1325))
        self.soc.listen(3)
        self.soc.setblocking(1)
        self.conn, self.addr = self.soc.accept()
        print('motor connected')
        
    def steer(self,prediction):
        if prediction == 2:
            self.conn.send(b'Forward')
            print('forward')
        elif prediction == 0:
            self.conn.send(b'Forward-Left')
            print("forward-left")
        elif prediction == 1:
            self.conn.send(b'Forward-Right')
            print("forward-right")
        else:
            self.conn.send(b'Forward')
    def stop(self):
        self.conn.send(b'Stop')
    def voice(self,check_point):
        if check_point == 1:
            self.conn.send(b'voice1')
        elif check_point == 2:
            self.conn.send(b'voice2')
        

class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.907778117067
        self.ay = 332.21335070570456

    def calculate(self, v, h, x_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            cv2.putText(image, "%.1fcm" % d,
                (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d


class ObjectDetection(object):

    def __init__(self):
        self.red_light = False
        self.green_light = False
        self.yellow_light = False

    def detect(self, cascade_classifier, gray_image, image):

        # y camera coordinate of the target point 'P'
        v = 0

        # minimum value to proceed traffic light state validation
        threshold = 100     
        
        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags=cv2.CASCADE_SCALE_IMAGE
        )

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5
            #print(x_pos+5, y_pos+5, x_pos+width-5, y_pos+height-5, width, height)

            # stop sign
            if width/height == 1:
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # traffic lights
            else:
                roi = gray_image[y_pos+10:y_pos + height-10, x_pos+10:x_pos + width-10]
                mask = cv2.GaussianBlur(roi, (25, 25), 0)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                
                # check if light is on
                if maxVal - minVal > threshold:
                    cv2.circle(roi, maxLoc, 5, (255, 0, 0), 2)
                    
                    # Red light
                    if 1.0/8*(height-30) < maxLoc[1] < 4.0/8*(height-30):
                        cv2.putText(image, 'Red', (x_pos+5, y_pos-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        self.red_light = True
                    
                    # Green light
                    elif 5.5/8*(height-30) < maxLoc[1] < (height-30):
                        cv2.putText(image, 'Green', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.green_light = True
    
                    # yellow light
                    elif 4.0/8*(height-30) < maxLoc[1] < 5.5/8*(height-30):
                        cv2.putText(image, 'Station', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        self.yellow_light = True
        return v


class SensorDataHandler(socketserver.BaseRequestHandler):

    data = " "

    def handle(self):
        global sensor_data
        try:
            while self.data:
                self.data = self.request.recv(1024)
                sensor_data = round(float(self.data[:10]), 1)
                #print "{} sent:".format(self.client_address[0])
                print(sensor_data)
        finally:
            print("Connection closed on thread 2")


class VideoStreamHandler(socketserver.StreamRequestHandler):

    # h1: stop sign
    h1 = 15.5 - 10  # cm
    # h2: traffic light
    h2 = 15.5 - 10
    
    nn = NeuralNetwork()
    nn.load_model("C:\\Users\\jbose\\AutoRCCar\\computer\\mlp_xml\\mlp_3.xml")
    obj_detection = ObjectDetection()
    rc_car = RCControl()

    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')
    light_cascade = cv2.CascadeClassifier('cascade_xml/traffic_light.xml')
    print('classifier loaded')
    
    d_to_camera = DistanceToCamera()
    d_stop_sign = 25
    d_light = 30

    stop_start = 0              # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    term_start = 0
    term_finish = 0
    term_time = 0
    drive_time_after_stop = 0
    

    def handle(self):
        print('stream...')
        global sensor_data
        stream_bytes = b' '
        stop_flag = False
        stop_sign_active = True
        
        yellow_active = True
        voice1_sent = False
        voice2_sent = False
        check_point = 1
        yellow_detected = False
        time_for_tts = 10
        term_for_next = False
        term_flag = False

        # stream video frames one by one
        try:
            print('stream start')
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                    # lower half of the image
                    half_gray = gray[120:240, :]

                    # object detection
                    v_param1 = self.obj_detection.detect(self.stop_cascade, gray, image)
                    v_param2 = self.obj_detection.detect(self.light_cascade, gray, image)

                    # distance measurement
                    if v_param1 > 0 or v_param2 > 0:
                        d1 = self.d_to_camera.calculate(v_param1, self.h1, 300, image)
                        d2 = self.d_to_camera.calculate(v_param2, self.h2, 100, image)
                        self.d_stop_sign = d1
                        self.d_light = d2

                    cv2.imshow('image', image)
                    #cv2.imshow('mlp_image', half_gray)

                    # reshape image
                    image_array = half_gray.reshape(1,38400).astype(np.float32)

                    # neural network makes prediction
                    prediction = self.nn.predict(image_array)
                    #prediction = 2
                    
                    # stop conditions
                    if sensor_data is not None and sensor_data < 25:
                        print("Stop, obstacle in front")
                        self.rc_car.stop() 
                    
                    elif 0 < self.d_stop_sign < 24 and stop_sign_active:
                        print("Stop sign ahead")
                        self.rc_car.stop()

                        # stop for 5 seconds
                        if stop_flag is False:
                            self.stop_start = cv2.getTickCount()
                            stop_flag = True
                        self.stop_finish = cv2.getTickCount()

                        self.stop_time = (self.stop_finish - self.stop_start)/cv2.getTickFrequency()
                        print("Stop time: %.2fs" % self.stop_time)

                        # 5 seconds later, continue driving
                        if self.stop_time > 5:
                            print("Waited for 5 seconds")
                            stop_flag = False
                            stop_sign_active = False
                            self.stop_time = 0
                    
                    elif self.d_light < 20:
                        #print("Traffic light ahead",self.d_light)
                        if self.obj_detection.red_light:
                            print("Red light")
                            self.rc_car.stop()
                        elif self.obj_detection.green_light:
                            print("Green light")
                        elif self.d_light < 13 and yellow_detected is False and self.obj_detection.yellow_light and not stop_flag and not term_flag:
                            print('Attraction sign detected')
                            print('please wait ',time_for_tts,' seconds for tts guiding')
                            self.rc_car.stop()
                            yellow_detected = True
                            yellow_active = False

                            if check_point is 1:
                                print('first tts command')
                                self.rc_car.voice(1)
                                check_point += 1
                            elif check_point is 2:
                                print('second tts command')
                                self.rc_car.voice(2)
                            if stop_flag is False:
                                self.stop_start = cv2.getTickCount()
                                stop_flag = True
                                
                        self.d_light = 30
                        self.obj_detection.red_light = False
                        self.obj_detection.green_light = False
                        self.obj_detection.yellow_light = False
                    elif term_flag:
                        self.term_finish = cv2.getTickCount()

                        self.term_time = (self.term_finish - self.term_start)/cv2.getTickFrequency()
                        #print('term_time : %.2fs'%self.term_time)
                        if self.term_time > 10:
                            term_flag = False

                        self.rc_car.steer(prediction)
                    elif stop_flag:
                        self.rc_car.stop()
                        
                        self.stop_finish = cv2.getTickCount()

                        self.stop_time = (self.stop_finish - self.stop_start)/cv2.getTickFrequency()
                        print('waiting time : %.2fs'% self.stop_time)

                        if self.stop_time > 10:
                            print('Waited for',time_for_tts,'seconds')
                            print('now begin driving')
                            stop_flag = False
                            self.stop_time = 0
                            yellow_detected = False
                            term_flag = True
                            self.term_start = cv2.getTickCount()
                                
                    else:
                        self.rc_car.steer(prediction)
                        self.stop_start = cv2.getTickCount()
                        self.d_stop_sign = 25

                        
                            
                        if stop_sign_active is False:
                            self.drive_time_after_stop = (self.stop_start - self.stop_finish)/cv2.getTickFrequency()
                            if self.drive_time_after_stop > 5:
                                stop_sign_active = True

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.rc_car.stop()
                        self.rc_car.conn.close()
                        cv2.destroyAllWindows()
                        sys.exit()
                        break

            cv2.destroyAllWindows()
            sys.exit()
        finally:
            print("Connection closed on thread 1")
            self.rc_car.stop()


class ThreadServer(object):

    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    def server_thread2(host, port):
        server = socketserver.TCPServer((host, port), SensorDataHandler)
        server.serve_forever()

    distance_thread = threading.Thread(target=server_thread2, args=(server_ip, 1326))
    distance_thread.start()
    video_thread = threading.Thread(target=server_thread(server_ip, 1324))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()
