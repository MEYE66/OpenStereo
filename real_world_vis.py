import numpy as np
import cv2


if __name__ == "__main__":
    root_path = "/home/lgz/dataset/ADEC/real/val/Test1/14_04_03_041"
    # left_img = np.load(f"{root_path}/left_rgb.npy")
    # right_img = np.load(f"{root_path}/right_rgb.npy")

    left_img = np.load(f"{root_path}/left_rectified.npy")
    right_img = np.load(f"{root_path}/right_rectified.npy")

    left_img = ((left_img - left_img.min()) / (left_img.max() - left_img.min()) * 255).astype(np.uint8)
    right_img = ((right_img - right_img.min()) / (right_img.max() - right_img.min()) * 255).astype(np.uint8)
    
    # left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    # right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"./left.png", left_img)
    cv2.imwrite(f"./right.png", right_img)