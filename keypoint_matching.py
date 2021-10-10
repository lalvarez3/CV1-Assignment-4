import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import DMatch as match_kp 
import scipy.ndimage as ndimage

def keypoint_matching(img1, img2):

    #first find keypoints and corresponding descriptors
    kp1,des1 = find_kp_des(img1)
    kp2,des2 = find_kp_des(img2)

    #match the kps/descriptors
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1,des2)

    #create the A matrix which will be used to perform affine transform
    A, b = create_Ab(matches,kp1,kp2)

    #one round of RANSAC (need to incorporate a loop)
    sampled_matches = np.random.choice(matches,10,replace = False)
    t_sp = fit_sample(sampled_matches,kp1,kp2)
    b_est = np.matmul(A,t_sp) 

    #unpack b_est, which are the estimated new coordinates of the keypoints
    #NOTE: I changed this cause when I debuged I noticed coordinates were swapped
    x_new = b_est[np.arange(0,len(b_est),2)]
    y_new = b_est[np.arange(1,len(b_est),2)]

    x_true = b[np.arange(0,len(b_est),2)]
    y_true = b[np.arange(1,len(b_est),2)]

    coords_new = list(zip(x_new,y_new))
    coords_true = list(zip(x_true,y_true))

    new_kps = cv2.KeyPoint_convert(coords_new)

    fake_matches = [match_kp(value, value, 0, 0.0) for value in range(len(kp1))]
    sampled_matches = np.random.choice(fake_matches,10,replace = False)
    img3 = cv2.drawMatches(img1,kp1,img2,new_kps,matches1to2=sampled_matches,outImg=None)
    # plt.imshow(img3)
    # plt.show()

    inliners_count = 0
    
    #calculate the number of inliners
    for new, true in zip(coords_new, coords_true):
        nx, ny = new
        tx, ty = true
        # TODO: we have to add neighbouthood of 10 pixels
        if int(nx) == int(tx) and int(ny) == int(ty):
            inliners_count += 1
    
    # PROBABLY WRONG CAUSE IT SAYS THEY ARE ALL INLINERS WHICH IS VERY UNLIKELY :(
    print(f'From a total of {len(coords_new)} objects, {inliners_count} are inliners')


    # cv2.circle(img2,)




def find_kp_des(img):
    gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp,des = sift.detectAndCompute(gray,None)
    return kp,des

def fit_sample(sampled_matches,kp1,kp2):
    # takes the sampled match indices as inputs, fits a line with least squares
    # to obtain the affine transformation parameters for 1 RANSAC sample
    A_sp,b_sp = create_Ab(sampled_matches,kp1,kp2)
    t_sp,*_ = np.linalg.lstsq(A_sp,b_sp)
    return t_sp

def create_Ab(matches,kp1,kp2):
    # given the matches and the keypoints, construct matrix A and vector b
    A = []
    b = []
    for x in matches:
        id1 = x.queryIdx
        id2 = x.trainIdx
        x1,y1 = kp1[id1].pt
        x2,y2 = kp2[id2].pt

        b.append(x2)
        b.append(y2)

        a = [[x1,y1,1,0,0,0],[0,0,0,x1,y1,1]]
        A.append(a)
    return np.concatenate(A), np.array(b)

    
if __name__ == '__main__':
    path_to_img1 = 'boat1.pgm'
    path_to_img2 = 'boat2.pgm'
    img1 = cv2.imread(path_to_img1)
    img2 = cv2.imread(path_to_img2)
    keypoint_matching(img1, img2)