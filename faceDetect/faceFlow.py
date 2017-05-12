#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from video import create_capture
from common import clock, draw_str
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils


camera = PiCamera()
camera.resolution = (240, 180)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(240, 180))
frame = rawCapture.array

rawCapture.truncate(0)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

#initialize VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video1 = cv2.VideoWriter('faceTrack.avi',fourcc, 20.0, (240,180))

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(10, 10),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects



def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    
class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        
    def run(self):
        cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        nested = cv2.CascadeClassifier('haarcascade_eye.xml')
        
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            img = frame.array
            vis = img.copy()
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blank = np.zeros_like(img)
            
            rects = detect(gray, cascade)
            draw_rects(blank, rects, (255))
            draw_rects(vis, rects, (0, 255, 0))

            gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
            
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                
                good = d < 1
                new_tracks = []
                
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                        
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 0, 255), thickness = 3)

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(gray, mask = mask, **feature_params)
                
                if p is not None:
                   for x, y in np.float32(p).reshape(-1, 2):
                      self.tracks.append([(x, y)])
                      
            self.frame_idx += 1
            self.prev_gray = gray
                
            if not nested.empty():
                for x1, y1, x2, y2 in rects:
                    roi = gray[y1:y2, x1:x2]
                    vis_roi = vis[y1:y2, x1:x2]
                    subrects = detect(roi.copy(), nested)
                    draw_rects(vis_roi, subrects, (255, 0, 0))

            cv2.imshow('facedetect', vis)
            #cv2.imshow('faceFlow', blank)
            #cv2.imshow('gray', gray)
            video1.write(vis)
            video1.write(vis)
            video1.write(vis)
            
            rawCapture.truncate(0)

            if cv2.waitKey(5) == 27:
                video1.release()
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)
    
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()
