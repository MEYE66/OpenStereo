import torch
import enum
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os



class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)



class Debayer5x5(torch.nn.Module):
    """Demosaicing of Bayer images using Malver-He-Cutler algorithm.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.

    Compared to Debayer2x2 this method does not use upsampling.
    Compared to Debayer3x3 the algorithm gives sharper edges and
    less chromatic effects.

    ## References
    Malvar, Henrique S., Li-wei He, and Ross Cutler.
    "High-quality linear interpolation for demosaicing of Bayer-patterned
    color images." 2004
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer5x5, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    # G at R,B locations
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [-2,  4,  8,  4, -2], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R,B at G in R rows
                    # scaled by 16
                    [ 0,  0,  1,  0,  0], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [-2,  8, 10,  8, -2], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [ 0,  0,  1,  0,  0], # noqa

                    # R,B at G in B rows
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 1,  0, 10,  0,  1], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R at B and B at R
                    # scaled by 16
                    [ 0,  0, -3,  0,  0], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [-3,  0, 12,  0, -3], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [ 0,  0, -3,  0,  0], # noqa

                    # R at R, B at B, G at G
                    # identity kernel not shown
                ]
            ).view(4, 1, 5, 5).float() / 16.0,
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            # Below, note that index 4 corresponds to identity kernel
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape

        xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        planes = torch.cat(
            (planes, x), 1
        )  # Concat with input to give identity kernel Bx5xHxW
        rgb = torch.gather(
            planes,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand for singleton batch dimension is faster
        )
        return torch.clamp(rgb, 0, 1)

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 1],  # pixel is R,G1
                [2, 3],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [3, 2],  # pixel is R,G1
                [1, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)




def convert_to_32bit_bayer_rg24_2(raw_data, height, width):
    # raw_data = np.fromfile(raw_data, dtype=np.uint8)
    raw_data = raw_data.reshape(-1, 3)
    raw_int32 = (raw_data[:, 0].astype(np.uint32) +
                (raw_data[:, 1].astype(np.uint32) << 8) +
                (raw_data[:, 2].astype(np.uint32) << 16))
    return raw_int32.reshape(height, width)


# Bilateral filter 
def bilateralFilter(bgr_image: np.ndarray) -> np.ndarray:
    # Bilateral filter to reduce grid-like artifacts while preserving edges
    filtered_image = cv2.bilateralFilter(bgr_image, d=1, sigmaColor=20, sigmaSpace=20)
    return filtered_image

def minmax_norm(image: np.ndarray) -> np.ndarray:
    # Min-max normalization to [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = np.zeros_like(image)  # Avoid division by zero
    return normalized_image.astype(np.float32)



def apply_gtm(img, eps=1e-6, param=0.18):
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = minmax_norm(img)
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out = minmax_norm(out)
    out = np.clip(out, 0, 1.).astype(np.float32)
    return out



def hdr_processing(raw_path, height, width):
    raw_hdr = np.load(raw_path)
    raw_hdr = convert_to_32bit_bayer_rg24_2(raw_hdr, height=height, width=width).astype(np.float32)
    # print(f"Loaded HDR image with shape {raw_hdr.shape} and range {raw_hdr.min()}-{raw_hdr.max()}")

    raw_hdr = minmax_norm(raw_hdr)

    debayer = Debayer5x5(layout=Layout.RGGB)  # Assuming RGGB layout
    rgb_image = debayer(torch.from_numpy(raw_hdr).unsqueeze(0).unsqueeze(0)).squeeze().numpy().transpose(1, 2, 0)

    rgb_image = bilateralFilter(rgb_image)
    rgb_image = minmax_norm(rgb_image).astype(np.float32)

    return rgb_image


def _process_single_folder(folder_path: Path, height=928, width=1440, overwrite=False):
    left_path = folder_path / "left.npy"
    right_path = folder_path / "right.npy"
    left_out = folder_path / "left_hdr.npy"
    right_out = folder_path / "right_hdr.npy"

    if not left_path.exists() or not right_path.exists():
        return "skip", str(folder_path), "missing left.npy/right.npy"

    if (not overwrite) and left_out.exists() and right_out.exists():
        return "skip", str(folder_path), "already exists"

    try:
        left_hdr = hdr_processing(left_path, height=height, width=width)
        right_hdr = hdr_processing(right_path, height=height, width=width)
        np.save(left_out, left_hdr)
        np.save(right_out, right_hdr)
        return "ok", str(folder_path), "saved left_hdr.npy/right_hdr.npy"
    except Exception as e:
        return "error", str(folder_path), str(e)


def preprocess_hdr_in_directory(root_dir,
                                height=928,
                                width=1440,
                                max_workers=8,
                                overwrite=False):
    """Multithread preprocessing for all folders containing left.npy/right.npy.

    Results are saved in-place as left_hdr.npy and right_hdr.npy.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    task_folders = []
    for dirpath, _, filenames in os.walk(root):
        names = set(filenames)
        if "left.npy" in names and "right.npy" in names:
            task_folders.append(Path(dirpath))

    if len(task_folders) == 0:
        print(f"No folders with left.npy/right.npy found under: {root_dir}")
        return {
            "total": 0,
            "ok": 0,
            "skip": 0,
            "error": 0,
        }

    print(f"Found {len(task_folders)} folders to process under: {root_dir}")
    print(f"Using max_workers={max_workers}, overwrite={overwrite}")

    ok_count = 0
    skip_count = 0
    err_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_single_folder,
                folder,
                height,
                width,
                overwrite,
            )
            for folder in task_folders
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            status, folder, msg = fut.result()
            if status == "ok":
                ok_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                err_count += 1
                print(f"[ERROR] {folder}: {msg}")

            if i % 50 == 0 or i == len(task_folders):
                print(f"Progress: {i}/{len(task_folders)} | ok={ok_count} skip={skip_count} error={err_count}")

    summary = {
        "total": len(task_folders),
        "ok": ok_count,
        "skip": skip_count,
        "error": err_count,
    }
    print(f"Done. Summary: {summary}")
    return summary



def img_vis_test():

    raw_path = "/home/lgz/dataset/ADEC/real/train/Scene1/17_58_49_761/left.npy"
    hdr_path = "/home/lgz/dataset/ADEC/real/train/Scene1/17_58_49_761/left_hdr.npy"
    height = 928
    width = 1440
    
    hdr_raw = hdr_processing(raw_path, height=height, width=width)

    hdr = np.load(hdr_path)


    hdr_raw = apply_gtm(minmax_norm(hdr_raw))
    hdr = apply_gtm(minmax_norm(hdr))

    cv2.imwrite("hdr_raw_vis.png", (hdr_raw * 255).astype(np.uint8))
    cv2.imwrite("hdr_vis.png", (hdr * 255).astype(np.uint8))
    cv2.imwrite("hdr_test.hdr", hdr.astype(np.float32))

    hdr_test = cv2.imread("hdr_test.hdr", cv2.IMREAD_UNCHANGED)
    hdr_test = minmax_norm(hdr_test)
    cv2.imwrite("hdr_test_vis.png", (hdr_test * 255).astype(np.uint8))

if __name__ == "__main__":
    import argparse


    img_vis_test()
    exit(234)
    parser = argparse.ArgumentParser(description="HDR preprocessing utility")
    parser.add_argument("--root_dir", type=str, default="/home/lgz/dataset/ADEC/real/train",
                        help="Root directory to scan recursively for left.npy/right.npy")
    parser.add_argument("--height", type=int, default=928, help="Raw image height")
    parser.add_argument("--width", type=int, default=1440, help="Raw image width")
    parser.add_argument("--workers", type=int, default=8, help="Number of threads")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing left_hdr.npy/right_hdr.npy")
    args = parser.parse_args()

    preprocess_hdr_in_directory(
        root_dir=args.root_dir,
        height=args.height,
        width=args.width,
        max_workers=args.workers,
        overwrite=args.overwrite,
    )

