import cv2


class VideoCaptureClass:
    def __init__(self, vid=0):
        self.vid = vid
        pass

    def __call__(self, *args, **kwds):
        self.cap = cv2.VideoCapture(self.vid)
        if not self.cap.isOpened():
            print("Не удалось открыть камеру!")

    def read_capture(self):
        success, frame = self.cap.read()
        return success, frame
