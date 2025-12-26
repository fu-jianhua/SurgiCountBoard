import cv2

def open_capture(source: str | int):
    if isinstance(source, str):
        s = source.strip()
        if s.isdigit():
            cap = cv2.VideoCapture(int(s))
        else:
            cap = cv2.VideoCapture(s)
    else:
        cap = cv2.VideoCapture(source)
    return cap