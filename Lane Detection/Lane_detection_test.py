import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=None, thickness=3):
    # If there are no lines to draw, exit.
    if color is None:
        color = [255, 0, 0]
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


def main():
    # image = cv.imread('img/test_img.png')
    image = cv2.imread('img/test_img4.png')
    # plt.figure()
    # plt.imshow(image)
    print('This image is:', type(image), 'with dimensions:', image.shape)

    width = int(image.shape[1])
    height = int(image.shape[0])

    region_of_interest_vertices = [
        (0, height),
        (0, height - 160),
        (width / 2, height / 2),
        (width, height - 160),
        (width, height),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(
        canny_image,
        np.array(
            [region_of_interest_vertices], np.int32), )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=100,
        lines=np.array([]),
        minLineLength=60,
        maxLineGap=25
    )
    line_image = draw_lines(image, lines)

    # cv2.imshow('frame1', image)
    # cv2.imshow('frame2', cropped_image)
    # cv2.waitKey(0)
    # image.release()
    # cv2.destroyAllWindows()

    plt.figure()
    plt.imshow(line_image)
    plt.show()


if __name__ == '__main__':
    main()
