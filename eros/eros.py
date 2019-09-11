import numpy as np
import cv2


def compute_mask(im_3D):
    assert isinstance(im_3D, np.ndarray), "Non-numpy input encountered."
    counts, bin = np.histogram(im_3D, bins=100)
    im_min, im_max = bin[5], bin[-5]

    threshold = im_min + 0.05*(im_max - im_min)
    return im_3D > threshold


def resample_at_angle(im_2D, phi, cent_of_rotation=None, compute_com_by_mask=None):
    row, col = im_2D.shape

    if cent_of_rotation is None:
        if not compute_com_by_mask is None:
            moments = cv2.moments(compute_com_by_mask)
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
        else:
            moments = cv2.moments(im_2D)
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
    else:
        cx, cy = cent_of_rotation

    A = cv2.getRotationMatrix2D((cx, cy), phi, 1)
    warpped_im_2D = cv2.warpAffine(im_2D, A, (row, col))
    return warpped_im_2D, cx, cy



def eros(im_3D, angular_res, angle_range=None):
    """
    Description
    -----------
        This function returns the best symmetry line angle that passes through the center of mass
        of the input image. The center of mass was calculated slice by slice.

    :param im_3D:       sitk.Image
    :param angular_res: (float|int)
    :param angle_range: tuple, default to None
    :return:
    """
    mask_3D = compute_mask(im_3D)
    im_3D[np.invert(mask_3D)] = 0

    out = []
    for z in range(len(im_3D)):
        # z = 24
        cx, cy = None, None
        row, col = im_3D[z].shape
        mid = col // 2
        if angle_range is None:
            angles = np.arange(180 // angular_res)*angular_res
        else:
            angles = np.linspace(angle_range[0], angle_range[1], (angle_range[1] - angle_range[0]) / angular_res)
        peak_scores = []
        for phi in angles:
            if cx is None:
                rot, cx, cy = resample_at_angle(im_3D[z], phi, compute_com_by_mask=mask_3D[z])
            else:
                rot, cx, cy = resample_at_angle(im_3D[z], phi, [cx, cy])

            sym_score_map = np.zeros_like(rot)
            for w in range(1, col-1):
                if w < mid:
                    N = (0, 2 * w)
                else:
                    N = (2*w - col, col)

                sub_sampled_rot = rot[:,N[0]:N[1]]
                sub_sampled_flipped_rot = np.fliplr(sub_sampled_rot)
                sym_score_map[:,w] = (np.sum(np.abs(sub_sampled_rot + sub_sampled_flipped_rot), axis=1) - np.sum(np.abs(sub_sampled_rot - sub_sampled_flipped_rot), axis=1)) / \
                            (np.sum(np.abs(sub_sampled_rot + sub_sampled_flipped_rot), axis=1) + np.sum(np.abs(sub_sampled_rot - sub_sampled_flipped_rot), axis=1))

            sym_score_map[np.isnan(sym_score_map)] = 0
            peak_score_per_row = np.max(sym_score_map, axis=1)
            peak_scores.append(np.sum(peak_score_per_row))

        best_angle = angles[np.argmax(peak_scores)]
        out.append([best_angle, (cx,cy)])
    return out

