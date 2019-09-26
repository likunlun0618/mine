import cv2
import numpy as np
import imageio


def read_video(file_name):
    images = []
    cap = cv2.VideoCapture(file_name)
    while True:
        ret, frame = cap.read()
        if ret:
            images.append(frame[np.newaxis, :])
        else:
            break
    cap.release()
    images = np.concatenate(images, axis=0)
    return images


def write_video(inp, out, fps=30):
    if type(inp) == list:
        h, w = cv2.imread(inp[0]).shape[:2]
        video_writer = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            fps,
            (w, h)
        )
        for name in inp:
            img = cv2.imread(name)
            video_writer.write(img)
    elif type(inp) == np.ndarray:
        h, w = inp.shape[1:3]
        video_writer = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            fps,
            (w, h)
        )
        for i in range(inp.shape[0]):
            video_writer.write(inp[i])
    else:
        assert(False)
