import numpy as np
import cv2
from scipy.interpolate import CubicSpline
from cone_detection import detect_cones


## This Function is fully implented by AI it's sole purpose is to generate the bird eye view 
## given the point from cone_detection/detect_cones



video_path = "data/Formula_test.mp4"

def transform_points(points, H):

    if len(points) == 0:
        return []

    points_np = np.float32(points).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(
        points_np,
        H
    )

    transformed = transformed.reshape(-1, 2)

    return [
        (int(x), int(y))
        for x, y in transformed
    ]


def point_inside_roi(point, roi):
    """ Check whether a point lies inside the quadrilateral ROI. """

    contour = roi.astype(np.float32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(
        contour,
        (float(point[0]), float(point[1])),
        False
    )

    return result >= 0


src_points = np.float32([
    [330, 350],
    [950, 350],
    [980, 480],
    [290, 480]
])

dst_points = np.float32([
    [0, 0],
    [600, 0],
    [600, 400],
    [0, 400]
])

H = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)


output_path = "bird_eye.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (600, 400)
)


frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    blue_points, yellow_points = detect_cones(frame)

    blue_points_filtered = [
        p for p in blue_points
        if point_inside_roi(p, src_points)
    ]

    yellow_points_filtered = [
        p for p in yellow_points
        if point_inside_roi(p, src_points)
    ]

    blue_ground = transform_points(
        blue_points_filtered,
        H
    )

    yellow_ground = transform_points(
        yellow_points_filtered,
        H
    )

    MIN_Y = 100

    left_cones = [
        p for p in blue_ground
        if p[1] >= MIN_Y
    ]

    right_cones = [
        p for p in yellow_ground
        if p[1] >= MIN_Y
    ]

    left_cones.sort(
        key=lambda p: p[1]
    )

    right_cones.sort(
        key=lambda p: p[1]
    )

    birdseye = np.zeros(
        (400, 600, 3),
        dtype=np.uint8
    )

    for x, y in left_cones:

        cv2.circle(
            birdseye,
            (x, y),
            8,
            (255, 0, 0),
            -1
        )

    for x, y in right_cones:

        cv2.circle(
            birdseye,
            (x, y),
            8,
            (0, 255, 255),
            -1
        )

    left_spline_x = None
    left_spline_y = None

    right_spline_x = None
    right_spline_y = None

    left_x = None
    left_y = None

    right_x = None
    right_y = None

    centerline = None

    if len(left_cones) >= 3:

        left_array = np.array(
            left_cones,
            dtype=float
        )

        left_array = left_array[
            np.argsort(left_array[:, 1])
        ]

        t = np.arange(
            len(left_array)
        )

        left_spline_x = CubicSpline(
            t,
            left_array[:, 0]
        )

        left_spline_y = CubicSpline(
            t,
            left_array[:, 1]
        )

        t_fine = np.linspace(
            0,
            len(left_array) - 1,
            200
        )

        left_x = left_spline_x(t_fine)
        left_y = left_spline_y(t_fine)

        left_points = np.column_stack(
            (
                left_x,
                left_y
            )
        ).astype(np.int32)

        valid = (
            (left_points[:, 0] >= 0) &
            (left_points[:, 0] < 600) &
            (left_points[:, 1] >= 0) &
            (left_points[:, 1] < 400)
        )

        left_points = left_points[valid]

        if len(left_points) > 1:

            cv2.polylines(
                birdseye,
                [left_points],
                False,
                (255, 0, 0),
                3
            )

    if len(right_cones) >= 3:

        right_array = np.array(
            right_cones,
            dtype=float
        )

        right_array = right_array[
            np.argsort(right_array[:, 1])
        ]

        t = np.arange(
            len(right_array)
        )

        right_spline_x = CubicSpline(
            t,
            right_array[:, 0]
        )

        right_spline_y = CubicSpline(
            t,
            right_array[:, 1]
        )

        t_fine = np.linspace(
            0,
            len(right_array) - 1,
            200
        )

        right_x = right_spline_x(t_fine)
        right_y = right_spline_y(t_fine)

        right_points = np.column_stack(
            (
                right_x,
                right_y
            )
        ).astype(np.int32)

        valid = (
            (right_points[:, 0] >= 0) &
            (right_points[:, 0] < 600) &
            (right_points[:, 1] >= 0) &
            (right_points[:, 1] < 400)
        )

        right_points = right_points[valid]

        if len(right_points) > 1:

            cv2.polylines(
                birdseye,
                [right_points],
                False,
                (0, 255, 255),
                3
            )

    if (
        left_spline_x is not None
        and right_spline_x is not None
    ):

        common_length = min(
            len(left_array),
            len(right_array)
        )

        if common_length >= 2:

            t_center = np.linspace(
                0,
                common_length - 1,
                200
            )

            center_left_x = left_spline_x(
                t_center
            )

            center_left_y = left_spline_y(
                t_center
            )

            center_right_x = right_spline_x(
                t_center
            )

            center_right_y = right_spline_y(
                t_center
            )

            center_x = (
                center_left_x +
                center_right_x
            ) / 2.0

            center_y = (
                center_left_y +
                center_right_y
            ) / 2.0

            centerline = np.column_stack(
                (
                    center_x,
                    center_y
                )
            )

    if centerline is not None:

        pts = centerline.astype(
            np.int32
        )

        valid = (
            (pts[:, 0] >= 0) &
            (pts[:, 0] < 600) &
            (pts[:, 1] >= 0) &
            (pts[:, 1] < 400)
        )

        pts = pts[valid]

        if len(pts) > 1:

            cv2.polylines(
                birdseye,
                [pts],
                False,
                (255, 255, 255),
                4
            )

    writer.write(
        birdseye
    )

    if frame_number % 100 == 0:

        print(
            f"Processed {frame_number} frames"
        )


cap.release()
writer.release()

print("\nDone!")
print("Output:", output_path)
