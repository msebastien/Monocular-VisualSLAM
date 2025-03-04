import cv2
import numpy as np
from display import Display
from pointmap import Map, Point
from extractor import Frame, add_ones, match_frames, denormalize_point

### Camera intrinsics
# Define principal point offset or optical center coordinates
W, H = 1920 // 2, 1080 // 2

# Define focus length
F = 270

# Define Intrinsic Matrix and inverse of that
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
Kinv = np.linalg.inv(K)

# Image display initialization
display = Display(W, H)

# Initialize a map
mapp = Map()
mapp.create_viewer()


def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    # Previous frame f2 to the current frame f1
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)

    # X_f1 = E * X_f2, f2 is in world coordinate frame, multiplying that with
    # [R|t] (Essential matrix) transforms the f2 pose with respect to the f1 coordinate frame
    f1.pose = np.dot(Rt, f2.pose)

    # The output is a matrix where each row is a 3D point
    # in homogeneous coordinates [X, Y, Z, W] returns an array of
    # size (n, 4), n = feature points
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])

    # The homogeneous coordinates [X, Y, Z, W] are converted to Euclidean coordinates
    pts4d /= pts4d[:, 3:]  # Divides each component X, Y, Z by the W component

    # Reject points without enough "Parallax" and points behind the camera
    # Returns a boolean array indicating which points satisfy both criteria
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
        # If the point is not looking good (good_pts4d[i] is False),
        # the loop skips the current iteration and moves to the next point
        if not good_pts4d[i]:
            continue

        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)

    # Denormalize point coordinates (between 0 and 1)
    # to pixel coordinates to display circles and lines on screen
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize_point(K, pt1)
        u2, v2 = denormalize_point(K, pt2)

        # Green circles (feature points)
        cv2.circle(img, (u1, v1), 3, (0, 255, 0))
        # Red lines (point matches between frames)
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))

    # 2D display
    display.paint()

    # 3D display
    mapp.display()


def triangulate(pose1, pose2, pts1, pts2):
    # The triangulation, used in epipolar geometry, relies
    # on image correspondences and camera calibration parameters.
    # It is used to get the 3D world coordinate of a 2D frame point
    # by projecting the point with the help of projection matrices
    # pose1: Pose matrix in 3D world homogeneous coordinates (Rotation and translation) for frame 1
    # P1 = K[I|0] (intrinsic, and essential matrix for extrinsic parameters)
    # pose2: Pose matrix in 3D world homogeneous coordinates for frame 2
    # P2 = K[R|t] (intrinsic, and essential matrix for extrinsic parameters)
    # Relationship between a 3D point X and its 2D projection x in an image
    # is given by the camera projection matrix P: x = PX
    # pts1: inlier feature points in frame 1 (2D frame coordinates)
    # pts2: inlier feature points in frame 2 (2D frame coordinates)

    # Initialize the result to store the homogeneous coordinates of the 3D points
    ret = np.zeros((pts1.shape[0], 4))

    # Invert the camera poses to get the projection matrices
    # It is necessary because the poses given are typically
    # transformations from the world coordinate frame (3D) to the camera coordinate frame (2D),
    # and we need the inverse to convert from camera (2D) to world coordinates (3D)
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)

    # Loop through each pair of corresponding points
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        # Initialize the matrix A (4x4) to hold the linear equations
        A = np.zeros((4, 4))

        # Populate the matrix A with the equations derived from
        # the projection matrices and the feature points
        # x1 = (u1, v1), x2 = (u2, v2) (2d camera frame points)
        # The cross product of 2 vectors x1 and x2 in a vector orthogonal to both, implying:
        # x1 * P1X = 0, with X = (X, Y, Z, W), the 3D world frame coordinate we want to get
        # x2 * P2X = 0
        # After expanding the cross product, we get:
        #   | u1P1(3) - P1(1) |  | X |
        #   | v1P1(3) - P1(2) |  | Y |  =  0, or Ax = 0
        #   | u2P2(3) - P2(1) |  | Z |
        #   | v2P2(3) - P2(2) |  | W |
        A[0] = p[0][0] * pose1[2] - pose1[0]  # u1 * P1(3) - P1(1)
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]

        # Perform SVD on A
        _, _, vt = np.linalg.svd(A)

        # The solution is the last row of V transposed (V^T),
        # corresponding to the smallest singular value
        ret[i] = vt[3]

    # Return the 3D points in homogeneous coordinates
    return ret


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
