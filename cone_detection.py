import numpy as np
import cv2


video_path = "data/Formula_test.mp4"

output_path_cone_detection = "Cone_output_video.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path_cone_detection,
    fourcc,
    fps,
    (width, height)
)


def detect_cones(frame2):

    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    LOWER_YELLOW = np.array([20, 125, 120])
    UPPER_YELLOW = np.array([30, 255, 255])

    LOWER_BLUE = np.array([94, 80, 2])
    UPPER_BLUE = np.array([140, 255, 255])

    LOWER_WHITE = np.array([0, 0, 115])
    UPPER_WHITE = np.array([179, 30, 255])

    blue_mask = cv2.inRange(
        hsv,
        LOWER_BLUE,
        UPPER_BLUE
    )

    white_mask = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )

    yellow_mask = cv2.inRange(
        hsv,
        LOWER_YELLOW,
        UPPER_YELLOW
    )

    blue_mask[:330, :] = 0
    white_mask[:330, :] = 0

    blue_mask[530:, 230:1050] = 0
    white_mask[530:, 230:1050] = 0

    blue_mask[430:, 400:800] = 0
    white_mask[430:, 400:800] = 0

    blue_white_mask = cv2.bitwise_or(
        blue_mask,
        white_mask
    )

    blue_white_contours, _ = cv2.findContours(
        blue_white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    yellow_contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blue_white_cones = []

    for contour in blue_white_contours:

        area = cv2.contourArea(contour)

        if area < 20 or area > 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = h / w

        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue

        blue_white_cones.append(
            (x, y, w, h)
        )

    yellow_cones = []

    for contour in yellow_contours:

        area = cv2.contourArea(contour)

        if area < 20 or area > 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = h / w

        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue

        yellow_cones.append(
            (x, y, w, h)
        )

    blue_white_points = []

    for x, y, w, h in blue_white_cones:

        cx = int(x + w / 2)
        cy = int(y + h)

        blue_white_points.append(
            (cx, cy)
        )

    yellow_points = []

    for x, y, w, h in yellow_cones:

        cx = int(x + w / 2)
        cy = int(y + h)

        yellow_points.append(
            (cx, cy)
        )

    return blue_white_points, yellow_points


while True:

    ret, frame = cap.read()

    if not ret:
        break

    blue_points, yellow_points = detect_cones(frame)

    for cx, cy in blue_points:
        cv2.circle(
            frame,
            (cx, cy),
            5,
            (255, 0, 0),
            -1
        )

    for cx, cy in yellow_points:
        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 255, 255),
            -1
        )

    out.write(frame)

cap.release()
out.release()

print("Finished!")
print(output_path_cone_detection)
