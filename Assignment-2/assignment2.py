import cv2
import numpy as np
import os


def load_image(image_path):
    """Helper function to safely load an image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    return img


def save_image(image, original_path, suffix):
    """Save processed image with suffix in the same folder"""
    folder = os.path.dirname(original_path)
    base = os.path.basename(original_path)
    name, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"
    out_path = os.path.join(folder, f"{name}_{suffix}{ext}")
    cv2.imwrite(out_path, image)
    print(f"Saved: {out_path}")


def padding(image, border_width):
    img = load_image(image)
    if img is None:
        return
    padded = cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width,
                                borderType=cv2.BORDER_REFLECT)
    save_image(padded, image, "padded")


def crop(image, x0, x1, y0, y1):
    img = load_image(image)
    if img is None:
        return
    h, w = img.shape[:2]
    cropped = img[y0:h - y1, x0:w - x1]
    save_image(cropped, image, "cropped")


def resize(image, width, height):
    img = load_image(image)
    if img is None:
        return
    resized = cv2.resize(img, (width, height))
    save_image(resized, image, "resized")


def copy(image, emptyPictureArray):
    img = load_image(image)
    if img is None:
        return
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            emptyPictureArray[i, j] = img[i, j]
    save_image(emptyPictureArray, image, "copied")


def grayscale(image):
    img = load_image(image)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_image(gray, image, "grayscale")


def hsv(image):
    img = load_image(image)
    if img is None:
        return
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    save_image(hsv_img, image, "hsv")


def hue_shifted(image, emptyPictureArray, hue):
    img = load_image(image)
    if img is None:
        return
    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                emptyPictureArray[i, j, k] = (int(img[i, j, k]) + hue) % 256
    save_image(emptyPictureArray, image, "hue_shifted")


def smoothing(image):
    img = load_image(image)
    if img is None:
        return
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    save_image(blurred, image, "smoothed")


def rotation(image, rotation_angle):
    img = load_image(image)
    if img is None:
        return

    if rotation_angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    else:
        print("Rotation angle must be 90 or 180")
        return
    save_image(rotated, image, f"rotated_{rotation_angle}")


if __name__ == "__main__":
    # Updated path
    image_path = r"D:\PROJECTS\UIA\MACHINE_VISION\Assignment-2\lena-2.png"

    padding(image_path, 100)
    crop(image_path, 80, 130, 80, 130)
    resize(image_path, 200, 200)

    img = load_image(image_path)
    if img is not None:
        h, w, c = img.shape
        empty = np.zeros((h, w, c), dtype=np.uint8)
        copy(image_path, empty)

        grayscale(image_path)
        hsv(image_path)

        empty2 = np.zeros((h, w, c), dtype=np.uint8)
        hue_shifted(image_path, empty2, 50)

        smoothing(image_path)
        rotation(image_path, 90)
        rotation(image_path, 180)
