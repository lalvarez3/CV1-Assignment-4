import cv2
import matplotlib.pyplot as plt
def keypoint_matching(img1, img2):

    # 1. Detect interest points in each image 

    # 2. Characterize the local appearance of the regions 
    # around interest points

    # 3. Get the set of matches between region descriptors in each image

    # 4. Perform RANSAC to discover the best transformation between images

    # The first 3 steps can be performed using David Lowe's SIFT
    # return keypoints

    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    img1=cv2.drawKeypoints(gray,kp,img1)

    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    img2=cv2.drawKeypoints(gray,kp,img2)
    print(des.shape)
    # plt.imshow(img1)
    # plt.show()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1,des2)
    # plt.imshow(img2)
    # plt.show()


    


if __name__ == '__main__':
    path_to_img1 = 'boat1.pgm'
    path_to_img2 = 'boat2.pgm'
    img1 = cv2.imread(path_to_img1)
    img2 = cv2.imread(path_to_img2)
    keypoint_matching(img1, img2)