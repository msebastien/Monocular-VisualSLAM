import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def add_ones(x):
    # creates homonogenous coordinates given the point x
    return np.concatenate([x, np.ones(x.shape[0], 1)], axis=1)


def normalize_points(Kinv, pts):
    # The inverse Intrinsic Matrix K^(-1) transforms 2D homogenous points
    # from pixel coordinates to normalized image coordinates.
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]


def denormalize_point(K, pt):
    # Converts a normalized point to pixel coordinates by
    # the Intrinsic Matrix and normalizing the result
    ret = np.dot(K, [pt[0], pt[1], 1.0])
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


def extract(img):
    # detects keypoints in an image, converts them to ORB keypoints,
    # then, computes ORB descriptors
    orb = cv2.ORB.create()

    # Detection
    pts = cv2.goodFeaturesToTrack(
        np.mean(img, axis=-1).astype(np.uint8),
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=10,
    )

    # Extraction
    kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
    kps, des = orb.compute(img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps])


def match_frames(f1, f2):
    # The code performs k-nearest neighbors matching on feature descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # applies Lowe's ratio test to filter out good
    # matches based on a distance threshold
    ret = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]

            # Distance test
            # additional distance test, ensuring that the
            # Euclidean distance between p1 and p2 is less than 0.1
            if np.linalg.norm((p1 - p2)) < 0.1:
                # Keep indices
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))

    assert len(ret) >= 8  # should have at least 8 matches between 2 frames
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Fit matrix
    # Compute the Fundamental Matrix using RANSAC
    model, inliers = ransac(
        ret[:, 0],
        ret[:, 1],
        FundamentalMatrixTransform,
        min_samples=8,
        residual_threshold=0.005,
        max_trials=200,
    )
    F = model.params

    # Ignore outliers
    ret = ret[inliers]

    # Notes:
    # OpenCV also allows to find R and t by decomposing the Essential matrix
    # using the cv2.recoverPose() method.
    # But, we need first to compute the Essential matrix using
    # the Intrinsic matrix and its inverse as well as the Fundamental matrix
    # Camera intrinsic parameters (replace with camera's calibration data)
    # K = np.array([[fx, 0, cx],
    #               [0, fy, cy],
    #               [0, 0, 1]])
    # Essential Matrix
    # E = K.T @ F @ K
    # Then, we can use OpenCV's builtin method.
    # _, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
    # R and t can be concatenated to get a single [R|t] matrix

    # Decompose directly the Fundamental matrix to get R and t
    Rt = extract_pose(F)

    return idx1[inliers], idx2[inliers], Rt


def extract_pose(F):
    # Define the W matrix for computing the Rotation matrix
    W = np.mat([0, -1, 0], [1, 0, 0], [0, 0, 1])

    # Perform Singular Value Decomposition (SVD) on the Fundamental matrix F
    U, d, Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0

    # Correct Vt if its determinant is negative
    # to ensure it's a proper rotation matrix
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Compute the initial rotation matrix R using U, W and Vt
    R = np.dot(np.dot(U, W), Vt)

    # Check the diagonal sum of R to ensure it's a proper rotation matrix
    # If not, recompute R using the transpose of W
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)

    # Extract the translation vector t from the third column of U
    t = U[:, 2]

    # Pose matrix [R|t]
    # Initialize a 4x4 identity matrix to store the pose
    ret = np.eye(4)
    # Set the top-left 3x3 submatrix to the rotation matrix R
    ret[:3, :3] = R
    # Set the top-right 3x1 submatrix to the translation vector t
    ret[:3, 3] = t

    print(d)

    # Return the 4x4 homogenous transformation matrix representing the pose
    return ret


class Frame(object):
    def __init__(self, mapp, img, K):
        # Intrinsic Matrix (used to determine P, the Camera Matrix)
        # Camera pose Matrix P = K[R|t], with [R|t] the Esssential (or Extrinsic) Matrix
        self.K = K
        self.Kinv = np.linalg.inv(self.K)  # Inverse of the Intrinsic Matrix

        # Initial pose of the frame (assuming IRt is predefined)
        # self.pose = IRt

        # Unique Frame ID based on the current number of frames in the map
        self.id = len(mapp.frames)

        # Extract feature points and descriptors from the image
        pts, self.des = extract(img)

        # Normalize the feature points using the inverse camera matrix
        self.pts = normalize_points(self.Kinv, pts)
