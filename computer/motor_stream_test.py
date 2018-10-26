import socketserver
import socket
import cv2
import numpy as np
import math
import os
import sys

class ObjectDetection(object):

    def __init__(self):
        self.red_light = False
        self.green_light = False
        self.yellow_light = False

    def detect(self, cascade_classifier, gray_image, image):

        # y camera coordinate of the target point 'P'
        v = 0

        # minimum value to proceed traffic light state validation
        threshold = 150     
        
        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
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
                    elif 5.5/8*(height-30) < maxLoc[1] < height-30:
                        cv2.putText(image, 'Green', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.green_light = True
    
                    # yellow light
                    elif 4.0/8*(height-30) < maxLoc[1] < 5.5/8*(height-30):
                        cv2.putText(image, 'Yellow', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        self.yellow_light = True
        return v


class RCControl(object):
    def __init__(self):
        self.soc = socket.socket()
        self.soc.bind(('192.168.0.10',1325))
        self.soc.listen(3)
        self.conn, self.addr = self.soc.accept()
        print('motor connected')
        
    def steer(self,prediction):
        if prediction == 2:
            self.conn.send(b'Forward')
            print('forward')
        elif prediction == 0:
            self.conn.send(b'Forward Left')
            print("forward-left")
        elif prediction == 1:
            self.conn.send(b'Forward Right')
            print("forward-right")
        else:
            self.conn.send(b'Stop')
    def stop(self):
        self.conn.send(b'Stop')
        
class VideoStreamHandler(socketserver.StreamRequestHandler):


    stop_cascade = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')
    light_cascade = cv2.CascadeClassifier('cascade_xml/traffic_light.xml')
    print('classifier loaded')

    rc_car = RCControl()

    d_stop_sign = 25
    d_light = 25

    stop_start = 0              # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    drive_time_after_stop = 0


    print('stream...')
    global sensor_data
    stream_bytes = b' '
    stop_flag = False
    stop_sign_active = True

    send_voice1 = False
    send_voice2 = False
    check_point = 1
    voice_response = False


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

                    # reshape image
                    image_array = half_gray.reshape(1,38400).astype(np.float32)

                    # neural network makes prediction
                    prediction = self.nn.predict(image_array)
                    
                    # stop conditions
                    
                    if 0 < self.d_stop_sign < 25 and stop_sign_active:
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
                    
        ##                    elif 0 < self.d_light < 10 and self.obj_detection.yellow_light:
        ##                        print('arrived at checkpoint')
        ##                        if check_point is 2:
        ##                            if not send_voice2:
        ##                                self.rc_car.voice('voice2')
        ##                                send_voice2 = True
        ##                            else:
        ##                                self.
        ##                                
        ##                        elif check_point is 1:
        ##                            if not send_voice1:
        ##                                self.rc_car.voice('voice1')
        ##                                send_voice1 = True

                    elif 0 < self.d_light < 30:
                        print("Traffic light ahead",self.d_light)
                        if self.obj_detection.red_light:
                            print("Red light")
                            self.rc_car.stop()
                        elif self.obj_detection.green_light:
                            print("Green light")
                            pass
                        elif self.obj_detection.yellow_light:
                            print("Yellow light flashing")
                            pass

                        self.d_light = 30
                        self.obj_detection.red_light = False
                        self.obj_detection.green_light = False
                        self.obj_detection.yellow_light = False

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
                        break

        finally:
            cv2.destroyAllWindows()
            sys.exit()


def server_thread(host, port):
    server = socketserver.TCPServer((host, port), VideoStreamHandler)
    server.serve_forever()
    
video_thread = threading.Thread(target=server_thread('192.168.0.10', 1324))
video_thread.start()

