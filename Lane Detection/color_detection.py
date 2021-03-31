import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255, 0, 0)
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def corner_detection():
    img = cv2.imread('img/test_img.png')
    img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

    # for i in range(len(corners)):
    #     for j in range(i + 1, len(corners)):
    #         corner1 = tuple(corners[i][0])
    #         corner2 = tuple(corners[j][0])
    #         color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
    #         cv2.line(img, corner1, corner2, color, 1)

    cv2.imshow('Frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image = cv2.imread('img/test_img.png')
    image2 = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    width = int(image.shape[1])
    height = int(image.shape[0])
    print(width, height)

    lower_gray = np.array([135, 135, 135], dtype="uint8")
    upper_gray = np.array([145, 145, 145], dtype="uint8")

    lower_yellow = np.array([0, 100, 100], dtype="uint8")
    upper_yellow = np.array([60, 255, 255], dtype="uint8")

    lower_black = np.array([0, 0, 0], dtype="uint8")
    upper_black = np.array([10, 10, 10], dtype="uint8")

    # region_of_interest_vertices = [
    #     (0, height),
    #     (0, height - 160),
    #     (width / 2, height / 2 + 10),
    #     (width, height - 160),
    #     (width, height),
    # ]
    #
    # cropped_image = region_of_interest(
    #     image,
    #     np.array(
    #         [region_of_interest_vertices], np.int32), )

    mask_gray = cv2.inRange(image, lower_gray, upper_gray)
    mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
    mask_black = cv2.inRange(image, lower_black, upper_black)

    output_gray = cv2.bitwise_and(image, image, mask=mask_gray)
    output_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)
    output_black = cv2.bitwise_and(image, image, mask=mask_black)

    gray_contours = cv2.findContours(mask_gray.copy(),
                                     cv2.RETR_CCOMP,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(image, gray_contours, -1, (0, 255, 0), 1)

    yellow_contours = cv2.findContours(mask_yellow.copy(),
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)[-2]
    cv2.drawContours(image, yellow_contours, -1, (255, 0, 0), 1)

    black_contours = cv2.findContours(mask_black.copy(),
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)[-2]
    cv2.drawContours(image, black_contours, -1, (0, 0, 255), 1)

    canny_gray = cv2.Canny(output_gray, 100, 200)
    canny_yellow = cv2.Canny(output_yellow, 100, 200)

    cv2.imshow("image", image)
    #cv2.imshow("image2", image2)
    # cv2.imshow('mash', mask_yellow)

    # cv.imshow("mask_black", mask_black)
    # cv.imshow("output", np.hstack([output1, output2]))
    # cv.imshow("canny", np.hstack([canny_gray, canny_yellow]))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plt.figure()
    # plt.imshow(image)
    # plt.show()


if __name__ == '__main__':
    main()
