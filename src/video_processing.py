import cv2
import matplotlib.pyplot as plt
import numpy as np

# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
def extract_video_frames(path: str = None) -> list:
    """
    Extracts frames from a video, and
    returns a list containing each frame.

    Inputs:
    -------

    path: str | None
        Path to the video.

    Outputs:
    -------
    frames: list
        List containing the frames of the video.

    """

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    frames = []
    count = 0
    while True:
        success, image = vidcap.read()
        if success:
            frames += [image]
            count += 1
            continue
        break

    return np.stack(frames, 0)

def label_frames(frames, frame_idx: list = None):
    """
    Function to use to label frames.

    Inputs:
    ------
    frames_idx: list | None
        Index of the frame to label.

    Outputs:
    -------
    label: list
        Labels of the chosen frame.
    """

    plt.imshow(frames[frame_idx])
    plt.show(block=False)
    print(f"Est-ce que il y a un rat sur cet image?")
    print(f"-> Taper 0 si il n'y a pas un rat")
    print(f"-> Taper 1 si il est dans le labyrinthe")
    print(f"-> Taper 2 dans la boîte")

    label = input()
    while label not in ["0", "1", "2"]:
        label = input()

    plt.close()

    return int(label)
