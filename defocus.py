import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

rgb = cv2.imread('rgb.png')
depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
height, width = rgb.shape[:2]
print(height, width)

depth = depth.astype(np.float32) / 1000

f = 0.5
F = 0.3
v0 = 1
def get_r(v):
    return np.abs((v0 - v) / v * F)
ppr = height / np.tan(30/180*np.pi)

def defocus(img_rgb, img_r):
    ids = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing='ij'), axis=-1).reshape(-1, 1, 2)
    neighbors = np.stack(np.meshgrid(np.arange(11), np.arange(11), indexing='ij'), axis=-1).reshape(-1, 2) - 5
    neighbor_ids = ids + neighbors
    valid_id_mask = np.logical_and(
                        np.logical_and(neighbor_ids[..., 0] >= 0, neighbor_ids[..., 0] < height),
                        np.logical_and(neighbor_ids[..., 1] >= 0, neighbor_ids[..., 1] < width))
    dists = neighbor_ids - ids
    dists = dists[..., 0]**2 + dists[..., 1]**2
    weights = np.exp(- 0.5 * dists / (img_r.reshape(-1, 1) * ppr / 3)**2)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    neighbor_rgb = np.zeros((height*width, neighbors.shape[0], 3), dtype=np.float32)
    valid_ids = neighbor_ids[valid_id_mask]
    neighbor_rgb[valid_id_mask] = img_rgb[valid_ids[:, 0], valid_ids[:, 1]]
    img_defocus = (neighbor_rgb * weights[..., None]).sum(axis=1).reshape(height, width, 3)
    return img_defocus.astype(np.uint8)


f_step = 0.01
while True:
    image_r = get_r(np.divide(depth * f, depth-f))
    rgb_defocused = defocus(rgb, image_r)
    scale = 4
    cv2.imshow("rgb", cv2.resize(rgb, (width*scale, height*scale)))
    cv2.imshow("rgb_defocused", cv2.resize(rgb_defocused, (width*scale, height*scale)))
    key = cv2.waitKey(1)
    if key == ord('w'):
        f = min(f + f_step, 2)
    elif key == ord('s'):
        f = max(0.5, f - f_step)
    elif key == ord('q'):
        break
