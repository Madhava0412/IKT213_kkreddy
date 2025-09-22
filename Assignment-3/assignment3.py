import cv2
import numpy as np


def detect_edges(img_file, sobel_out="sobel_edges.jpg", canny_out="canny_edges.jpg",
                 canny_t1=50, canny_t2=50):
    picture = cv2.imread(img_file)
    gray_pic = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    blur_pic = cv2.GaussianBlur(gray_pic, (3, 3), 0)

    # --- Sobel ---
    sobel_xy = cv2.Sobel(blur_pic, cv2.CV_64F, 1, 1, ksize=1)
    sobel_xy = cv2.convertScaleAbs(sobel_xy)
    cv2.imwrite(sobel_out, sobel_xy)
    print(f"[INFO] Sobel edges saved -> {sobel_out}")

    # --- Canny ---
    canny_edges = cv2.Canny(blur_pic, canny_t1, canny_t2)
    cv2.imwrite(canny_out, canny_edges)
    print(f"[INFO] Canny edges saved -> {canny_out}")


def find_template(main_img, patch_img, result_file="template_match.jpg"):
    main = cv2.imread(main_img)
    main_gray = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY)
    patch = cv2.imread(patch_img, 0)
    w, h = patch.shape[::-1]

    score_map = cv2.matchTemplate(main_gray, patch, cv2.TM_CCOEFF_NORMED)
    matches = np.where(score_map >= 0.9)   # threshold = 0.9

    for loc in zip(*matches[::-1]):
        cv2.rectangle(main, loc, (loc[0] + w, loc[1] + h), (0, 0, 255), 2)

    cv2.imwrite(result_file, main)
    print(f"[INFO] Template match saved -> {result_file}")


def scale_image(img_file, steps=2, mode="up", result_file="resized.jpg"):
    picture = cv2.imread(img_file)

    if mode == "up":
        for _ in range(steps):
            picture = cv2.pyrUp(picture)
    elif mode == "down":
        for _ in range(steps):
            picture = cv2.pyrDown(picture)
    else:
        raise ValueError("mode must be 'up' or 'down'")

    cv2.imwrite(result_file, picture)
    print(f"[INFO] Resized image saved -> {result_file}")


if __name__ == "__main__":
    lambo = "lambo.png"
    shapes = "shapes-1.png"
    template = "shapes_template.jpg"

    detect_edges(lambo, sobel_out="sobel_edges.jpg", canny_out="canny_edges.jpg")
    find_template(shapes, template, result_file="template_match.jpg")
    scale_image(lambo, steps=2, mode="up", result_file="resized.jpg")
