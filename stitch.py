#takes an image pair as input, and return the stitched version

from keypoint_matching import *

def stitch(img1,img2,rounds = 50000):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    estimated_shape = (img1_gray.shape[0] + img2_gray.shape[0],img1_gray.shape[1] + img2_gray.shape[1])
    stitched_img = np.zeros(estimated_shape)
    large_img1 = to_shape(img1_gray,estimated_shape)
    large_img2 = to_shape(img2_gray,estimated_shape)

    _,Proj_est = RANSAC(large_img1,large_img2,rounds=rounds)

    [a,b,c,d,e,f] = Proj_est
    Aff_P = np.array([[a,b,c], [d, e, f], [0, 0, 1]])

    rows,columns = large_img1.shape[:2]
    for i in range(rows):
        for j in range(columns):
            x_n,y_n,_ = np.matmul(Aff_P,np.array([i,j,1]))
            value = large_img1[i,j]

            if x_n >= 0 and x_n < estimated_shape[0] and y_n >= 0 and y_n < estimated_shape[1]:
                stitched_img[int(x_n),int(y_n)] = value

    print(stitched_img)

    # blended = cv2.addWeighted(stitched_img.astype(np.float32), 0.5, large_img2.astype(np.float32), 0.5, 0)
    plt.imshow(stitched_img)
    # plt.imshow(stitched_img)
    plt.show()

    plt.imshow(large_img2)
    plt.show()


def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

#keypoints and matches
if __name__ == '__main__':

    path_to_img1 = 'boat1.pgm'
    path_to_img2 = 'boat2.pgm'
    # path_to_img1 = 'left.jpg'
    # path_to_img2 = 'right.jpg'

    img1 = cv2.imread(path_to_img1)
    img2 = cv2.imread(path_to_img2)

    stitch(img1,img2,rounds=100)


