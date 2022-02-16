"""
script to create videos from images
"""
# pylint: disable=E1101
import os
import cv2

DATE = "2017-12-09"  # name of the folder with images
IMAGE_FOLDER = '../videos/' + DATE  # folder where image-folders are stored
VIDEO_NAME = DATE + '.mp4'  # name of the mp4 file
VIDEO_FOLDER = "../videos/"  # folder where rendered videos are stored

if __name__ == '__main__':
    images = [img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(VIDEO_FOLDER + VIDEO_NAME, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(IMAGE_FOLDER, image)))

    cv2.destroyAllWindows()
    video.release()
