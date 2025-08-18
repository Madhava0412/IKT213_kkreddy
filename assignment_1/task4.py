import cv2
import numpy as np

def print_image_information(image):
    height, width, channels = image.shape
    print("Image information:")
    print("Width:", width)
    print("Height:", height)
    print("Channels:", channels)
    print("Size:", image.size)
    print("Data type:", image.dtype)


def main():
    image = cv2.imread(r"D:\PROJECTS\UIA\MACHINE_VISION\Assignment_1\lena-1.png")
    print_image_information(image)

if __name__ == "__main__":
    main()