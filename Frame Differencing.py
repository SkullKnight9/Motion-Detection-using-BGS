import cv2
import numpy as np

class FrameDifferencer:
    def __init__(self, scaling_factor=1.0):
        self.scaling_factor = scaling_factor

    def compute_frame_diff(self, prev_frame, cur_frame, next_frame):
        diff_frames1 = cv2.absdiff(next_frame, cur_frame)
        diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
        return cv2.bitwise_and(diff_frames1, diff_frames2)

    def get_frame(self, cap):
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=self.scaling_factor,
            fy=self.scaling_factor, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

if __name__=='__main__':
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"Media/Original Video.mpg")
    differencer = FrameDifferencer(scaling_factor=1.0)

    prev_frame = differencer.get_frame(cap)
    cur_frame = differencer.get_frame(cap)
    next_frame = differencer.get_frame(cap)

    while True:
        frame_difference = differencer.compute_frame_diff(prev_frame, cur_frame, next_frame)
        _, frame_th = cv2.threshold(frame_difference, 0, 255, cv2.THRESH_TRIANGLE)
        ret, frame = cap.read()

        cv2.imshow("Object Movement", frame_difference)
        cv2.imshow("Object Detection", frame_th)
        cv2.imshow("Original Video", frame)

        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = differencer.get_frame(cap)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
