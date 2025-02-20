import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def add_ones(x):
    # creates homonogenous coordinates given the point x
    return np.concatenate([x, np.ones(x.shape[0], 1)], axis=1)


def normalize_points(Kinv, pts):
    # The inverse camera matrix K^(-1) transforms 2D homogenous points
    # from pixel coordinates to normalized image coordinates.
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]


def denormalize_point(K, pt):
    # Converts a normalized point to pixel coordinates by
    # the Camera Matrix and normalizing the result
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
    model, inliers = ransac(
        ret[:, 0],
        ret[:, 1],
        FundamentalMatrixTransform,
        min_samples=8,
        residual_threshold=0.005,
        max_trials=200,
    )

    # Ignore outliers
    ret = ret[inliers]
    # Rt = extract_pose(model.params)

    return idx1[inliers], idx2[inliers]  # , Rt


class Frame(object):

    def __init__(self, mapp, img, K):
        self.K = K  # Camera Matrix (Intrinsic parameters)
        self.Kinv = np.linalg.inv(self.K)  # Inverse of the Camera Matrix
        # Initial pose of the frame (assuming IRt is predefined)
        # self.pose = IRt

        # Unique Frame ID based on the current number of frames in the map
        self.id = len(mapp.frames)

        # Extract feature points and descriptors from the image
        pts, self.des = extract(img)

        # Normalize the feature points using the inverse camera matrix
        self.pts = normalize_points(self.Kinv, pts)
