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
    kp1 = sift.detect(gray1,None)
    img1=cv2.drawKeypoints(gray1,kp1,img1)

    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp2, des = sift.detectAndCompute(gray2,None)
    img2=cv2.drawKeypoints(gray2,kp2,img2)
    print(des.shape)
    # plt.imshow(img1)
    # plt.show()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(gray1,gray2)
    plt.imshow(img2)
    # plt.show()


    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

    plt.imshow(img3),plt.show()


    


if __name__ == '__main__':
    path_to_img1 = 'boat1.pgm'
    path_to_img2 = 'boat2.pgm'
    img1 = cv2.imread(path_to_img1)
    img2 = cv2.imread(path_to_img2)
    keypoint_matching(img1, img2)