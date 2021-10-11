import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import DMatch as match_kp
import scipy.ndimage as ndimage
from sklearn.metrics.pairwise import paired_distances

np.random.seed(seed=0)

def RANSAC_round(matches, kp1, new_kp1, kp2, A, b):
    sampled_matches = np.random.choice(matches, 10, replace=False)
    t_sp = fit_sample(sampled_matches, kp1, kp2)
    b_est = np.matmul(A, t_sp)

    # unpack b_est, which are the estimated new coordinates of the keypoints
    x_new = b_est[np.arange(0, len(b_est), 2)]
    y_new = b_est[np.arange(1, len(b_est), 2)]

    x_true = b[np.arange(0, len(b_est), 2)]
    y_true = b[np.arange(1, len(b_est), 2)]

    coords_new = list(zip(x_new, y_new))
    coords_true = list(zip(x_true, y_true))

    new_kps = cv2.KeyPoint_convert(coords_new)

    subset = np.linspace(0, len(new_kp1)-1, 100, dtype=int)
    
    new_kps = np.array(new_kps)
    fake_matches = [match_kp(value, value, 0, 0.0)
                    for value in range(len(new_kp1[subset]))]
    #TODO: increase point size and reduce sample size
    img3 = cv2.drawMatches(
        img1, new_kp1[subset], img2, new_kps[subset], matches1to2=fake_matches, outImg=None)
    # plt.show()

    est_coordinates = np.stack([x_new, y_new], axis=1)
    true_coordinates = np.stack([x_true, y_true], axis=1)

    # TODO: we have to return the inliners
    distances = paired_distances(est_coordinates, true_coordinates)
    inliners = np.sum(distances <= 10)

    # print(
    #         f'From a total of {len(coords_new)} objects, {inliners} are inliners')
        

    return inliners, t_sp, new_kps


def keypoint_matching(img1, img2):

    # first find keypoints and corresponding descriptors
    kp1, des1 = find_kp_des(img1)
    kp2, des2 = find_kp_des(img2)

    # match the kps/descriptors
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)

    # create the A matrix which will be used to perform affine transform
    A, b, new_kp1, new_kp2 = create_Ab(matches, kp1, kp2)

    # one round of RANSAC (need to incorporate a loop)
    rounds = 1
    best_total_inliners = (0, [-1])

    for round in range(rounds):
        inliners, t_sp, new_kps = RANSAC_round(matches, kp1, new_kp1, kp2, A, b)
        # TODO: we also have to save the inliners (points)
        if inliners > best_total_inliners[0]:
            best_total_inliners = (inliners, t_sp, new_kps)
    
    print(f'Best result {best_total_inliners[0]} with t: {best_total_inliners[1]}')

    new_kps = best_total_inliners[2]
    subset = np.linspace(0, len(new_kp1)-1, 100, dtype=int)
    fake_matches = [match_kp(value, value, 0, 0.0)
            for value in range(len(new_kp1[subset]))]
    img3 = cv2.drawMatches(
    img1, new_kp1[subset], img2, new_kps[subset], matches1to2=fake_matches, outImg=None)        
    plt.imshow(img3)
    plt.savefig(f'best_result_{best_total_inliners[0]}_inliners')

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    new_A = []
    coordinates = []
    for row in gray:
        for column in row:
            x1 = row
            y1 =  column
            coordinates.append((x1, y1))
            new_A.append([[x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1]])
 
    new_A = np.concatenate(new_A)

    # shape of b_est is (1156000,)
    b_est = np.matmul(new_A, best_total_inliners[1])

    # TODO: shape after this gives (578000, 2, 850) for x_new
    x_new = b_est[np.arange(0, len(b_est), 2)]
    y_new = b_est[np.arange(1, len(b_est), 2)]

    # TODO: fix coordinates shape dont match
    coords_new = list(zip(x_new, y_new))
    coords_new_int = [ (int(x),int(y)) for x, y in coords_new]

    new_rotated_0 = np.zeros(np.shape(img1))
    new_rotated = np.full(np.shape(img1), fill_value = -1)

    for init, new in zip(coordinates, coords_new_int):
        pixel_value = img1[init[0], init[1]]
        new_rotated_0[new[0], new[1]] =  pixel_value
        new_rotated[new[0], new[1]] =  pixel_value

    plt.imshow(new_rotated_0)
    plt.savefig(f'new_rotated')

    



       
    

    # cv2.circle(img2,)


def find_kp_des(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def fit_sample(sampled_matches, kp1, kp2):
    # takes the sampled match indices as inputs, fits a line with least squares
    # to obtain the affine transformation parameters for 1 RANSAC sample
    A_sp, b_sp, new_kp1, new_kp2 = create_Ab(sampled_matches, kp1, kp2)
    t_sp, res, *_ = np.linalg.lstsq(A_sp, b_sp, rcond=-1)
    return t_sp


def create_Ab(matches, kp1, kp2):
    # given the matches and the keypoints, construct matrix A and vector b
    A = []
    b = []

    new_kp1 = []
    new_kp2 = []

    for x in matches:
        id1 = x.queryIdx
        id2 = x.trainIdx

        x1, y1 = kp1[id1].pt
        x2, y2 = kp2[id2].pt

        new_kp1.append(kp1[id1])
        new_kp2.append(kp2[id2])

        b.append(x2)
        b.append(y2)

        a = [[x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1]]
        A.append(a)

    new_kp1 = np.array(new_kp1)
    new_kp2 = np.array(new_kp2)
    
    return np.concatenate(A), np.array(b), new_kp1, new_kp2


if __name__ == '__main__':
    path_to_img1 = 'boat1.pgm'
    path_to_img2 = 'boat2.pgm'
    img1 = cv2.imread(path_to_img1)
    img2 = cv2.imread(path_to_img2)
    keypoint_matching(img1, img2)
