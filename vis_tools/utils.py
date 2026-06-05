import cv2
import numpy as np


def undistort_with_k1(img, k1, f_scale=1.0):
    """
    Apply single-parameter radial undistortion.

    Args:
        img: input image, H x W x C or H x W
        k1: radial distortion coefficient
        f_scale: virtual focal length scale

    Returns:
        undistorted image
    """

    h, w = img.shape[:2]

    cx = w / 2.0
    cy = h / 2.0

    # virtual focal length
    f = f_scale * max(h, w)

    # pixel grid in output image
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # normalize coordinates
    xn = (x - cx) / f
    yn = (y - cy) / f

    r2 = xn ** 2 + yn ** 2

    # inverse mapping:
    # output undistorted coordinate -> sample distorted coordinate
    scale = 1.0 + k1 * r2

    xd = xn * scale
    yd = yn * scale

    map_x = (xd * f + cx).astype(np.float32)
    map_y = (yd * f + cy).astype(np.float32)

    undistorted = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return undistorted

def line_straightness_score(img, canny_th1=80, canny_th2=160):
    """
    Estimate how straight the dominant lines are.
    Lower score means straighter lines.

    Args:
        img: input image

    Returns:
        score: straightness score
    """

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    gray = gray.astype(np.uint8)

    edges = cv2.Canny(gray, canny_th1, canny_th2)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min(img.shape[:2]) // 5,
        maxLineGap=20
    )

    if lines is None:
        return np.inf

    total_error = 0.0
    total_count = 0

    edge_points = np.column_stack(np.where(edges > 0))
    # edge_points: y, x

    for line in lines[:50]:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        length = np.sqrt(dx * dx + dy * dy)
        if length < 1:
            continue

        # line equation: ax + by + c = 0
        a = dy
        b = -dx
        c = dx * y1 - dy * x1

        xs = edge_points[:, 1]
        ys = edge_points[:, 0]

        dist = np.abs(a * xs + b * ys + c) / np.sqrt(a * a + b * b)

        # only count edge points close to this line
        nearby = dist < 3.0

        if np.sum(nearby) < 20:
            continue

        error = np.mean(dist[nearby])
        total_error += error
        total_count += 1

    if total_count == 0:
        return np.inf

    return total_error / total_count

def blind_radial_undistort(
    img,
    k1_range=(-5, 5),
    num_steps=121,
    f_scale=1.0,
    resize_for_search=0.5
):
    """
    Blind radial distortion correction without calibration.

    Args:
        img: input RGB or grayscale image
        k1_range: search range for radial distortion coefficient
        num_steps: number of k1 candidates
        f_scale: virtual focal length scale
        resize_for_search: resize image for faster parameter search

    Returns:
        best_img: undistorted image
        best_k1: estimated k1
        scores: list of (k1, score)
    """

    h, w = img.shape[:2]

    if resize_for_search != 1.0:
        small = cv2.resize(
            img,
            None,
            fx=resize_for_search,
            fy=resize_for_search,
            interpolation=cv2.INTER_AREA
        )
    else:
        small = img.copy()

    k1_values = np.linspace(k1_range[0], k1_range[1], num_steps)

    best_score = np.inf
    best_k1 = 0.0
    scores = []

    for k1 in k1_values:
        candidate = undistort_with_k1(
            small,
            k1=k1,
            f_scale=f_scale
        )

        score = line_straightness_score(candidate)

        scores.append((k1, score))

        if score < best_score:
            best_score = score
            best_k1 = k1

    best_img = undistort_with_k1(
        img,
        k1=best_k1,
        f_scale=f_scale
    )

    return best_img



def undistort_radial_bgr(img_bgr, k1=-0.32, k2=0.08, p1=0.0, p2=0.0, k3=0.0, f_scale=0.72, alpha=0.0):
    h, w = img_bgr.shape[:2]
    fx = fy = f_scale * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    out = cv2.undistort(img_bgr, K, D, None, newK)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0 and alpha == 0.0:
        # Resize cropped valid area back to original size to avoid black borders.
        out = out[y:y+rh, x:x+rw]
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out

# Barrel-like correction: negative k1 in OpenCV distortion model generally pulls edges outward after undistortion.
# corrected_bgr = undistort_radial_bgr(img_bgr, k1=-0.34, k2=0.10, f_scale=0.70, alpha=0.0)