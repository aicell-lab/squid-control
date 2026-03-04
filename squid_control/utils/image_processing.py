"""
Created on Mon May  7 19:44:40 2018

@author: Francois and Deepak
"""

import cv2
import numpy as np
from numpy import mean, square

# color is a vector HSV whose size is 3


def default_lower_HSV(color):
    c = [0, 100, 100]
    c[0] = np.max([color[0] - 10, 0])
    c[1] = np.max([color[1] - 40, 0])
    c[2] = np.max([color[2] - 40, 0])
    return np.array(c, dtype="uint8")


def default_upper_HSV(color):
    c = [0, 255, 255]
    c[0] = np.min([color[0] + 10, 178])
    c[1] = np.min([color[1] + 40, 255])
    c[2] = np.min([color[2] + 40, 255])
    return np.array(c, dtype="uint8")


def threshold_image(image_BGR, LOWER, UPPER):
    image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
    imgMask = 255 * np.array(
        cv2.inRange(image_HSV, LOWER, UPPER), dtype="uint8"
    )  # The tracked object will be in white
    imgMask = cv2.erode(
        imgMask, None, iterations=2
    )  # Do a series of erosions and dilations on the thresholded image to reduce smaller blobs
    imgMask = cv2.dilate(imgMask, None, iterations=2)

    return imgMask


def threshold_image_gray(image_gray, LOWER, UPPER):
    imgMask = np.array((image_gray >= LOWER) & (image_gray <= UPPER), dtype="uint8")

    # imgMask = cv2.inRange(cv2.UMat(image_gray), LOWER, UPPER)  #The tracked object will be in white
    imgMask = cv2.erode(
        imgMask, None, iterations=2
    )  # Do a series of erosions and dilations on the thresholded image to reduce smaller blobs
    imgMask = cv2.dilate(imgMask, None, iterations=2)

    return imgMask


def bgr2gray(image_BGR):
    return cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)


def crop(image, center, imSize):  # center is the vector [x,y]
    imH, imW, *rest = (
        image.shape
    )  # image.shape:[nb of row -->height,nb of column --> Width]
    xmin = max(10, center[0] - int(imSize))
    xmax = min(imW - 10, center[0] + int(imSize))
    ymin = max(10, center[1] - int(imSize))
    ymax = min(imH - 10, center[1] + int(imSize))
    return np.array([[xmin, ymin], [xmax, ymax]]), np.array(image[ymin:ymax, xmin:xmax])


def crop_image(image, crop_width, crop_height):
    image_height = image.shape[0]
    image_width = image.shape[1]
    roi_left = int(max(image_width / 2 - crop_width / 2, 0))
    roi_right = int(min(image_width / 2 + crop_width / 2, image_width))
    roi_top = int(max(image_height / 2 - crop_height / 2, 0))
    roi_bottom = int(min(image_height / 2 + crop_height / 2, image_height))
    image_cropped = image[roi_top:roi_bottom, roi_left:roi_right]
    image_cropped_height = image_cropped.shape[0]
    image_cropped_width = image_cropped.shape[1]
    return image_cropped, image_cropped_width, image_cropped_height


def get_bbox(cnt):
    return cv2.boundingRect(cnt)


def find_centroid_enhanced(image, last_centroid):
    # find contour takes image with 8 bit int and only one channel
    # find contour looks for white object on a black back ground
    # This looks for all contours in the thresholded image and then finds the centroid that maximizes a tracking metric
    # Tracking metric : current centroid area/(1 + dist_to_prev_centroid**2)
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    centroid = False
    isCentroidFound = False
    if len(contours) > 0:
        all_centroid = []
        dist = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = np.array([cx, cy])
                isCentroidFound = True
                all_centroid.append(centroid)
                dist.append(
                    [cv2.contourArea(cnt) / (1 + (centroid - last_centroid) ** 2)]
                )

    if isCentroidFound:
        ind = dist.index(max(dist))
        centroid = all_centroid[ind]

    return isCentroidFound, centroid


def find_centroid_enhanced_Rect(image, last_centroid):
    # find contour takes image with 8 bit int and only one channel
    # find contour looks for white object on a black back ground
    # This looks for all contours in the thresholded image and then finds the centroid that maximizes a tracking metric
    # Tracking metric : current centroid area/(1 + dist_to_prev_centroid**2)
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    centroid = False
    isCentroidFound = False
    rect = False
    if len(contours) > 0:
        all_centroid = []
        dist = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = np.array([cx, cy])
                isCentroidFound = True
                all_centroid.append(centroid)
                dist.append(
                    [cv2.contourArea(cnt) / (1 + (centroid - last_centroid) ** 2)]
                )

    if isCentroidFound:
        ind = dist.index(max(dist))
        centroid = all_centroid[ind]
        cnt = contours[ind]
        xmin, ymin, width, height = cv2.boundingRect(cnt)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        imH, imW = image.shape
        width = min(width, imW - int(cx))
        height = min(height, imH - int(cy))
        rect = (xmin, ymin, width, height)

    return isCentroidFound, centroid, rect


def find_centroid_basic(image):
    # find contour takes image with 8 bit int and only one channel
    # find contour looks for white object on a black back ground
    # This finds the centroid with the maximum area in the current frame
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    centroid = False
    isCentroidFound = False
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = np.array([cx, cy])
            isCentroidFound = True
    return isCentroidFound, centroid


def find_centroid_basic_Rect(image):
    # find contour takes image with 8 bit int and only one channel
    # find contour looks for white object on a black back ground
    # This finds the centroid with the maximum area in the current frame and alsio the bounding rectangle. - DK 2018_12_12
    imH, imW = image.shape
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    centroid = False
    isCentroidFound = False
    bbox = None
    rect = False
    if len(contours) > 0:
        # Find contour with max area
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            # Centroid coordinates
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = np.array([cx, cy])
            isCentroidFound = True

            # Find the bounding rectangle
            xmin, ymin, width, height = cv2.boundingRect(cnt)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            width = min(width, imW - xmin)
            height = min(height, imH - ymin)

            bbox = (xmin, ymin, width, height)

    return isCentroidFound, centroid, bbox


def scale_square_bbox(bbox, scale_factor, square=True):

    xmin, ymin, width, height = bbox

    if square == True:
        min_dim = min(width, height)
        width, height = min_dim, min_dim

    new_width, new_height = int(scale_factor * width), int(scale_factor * height)

    new_xmin = xmin - (new_width - width) / 2
    new_ymin = ymin - (new_height - height) / 2

    new_bbox = (new_xmin, new_ymin, new_width, new_height)
    return new_bbox


def get_image_center_width(image):
    ImShape = image.shape
    ImH, ImW = ImShape[0], ImShape[1]
    return np.array([ImW * 0.5, ImH * 0.5]), ImW


def get_image_height_width(image):
    ImShape = image.shape
    ImH, ImW = ImShape[0], ImShape[1]
    return ImH, ImW


def get_image_top_center_width(image):
    ImShape = image.shape
    ImH, ImW = ImShape[0], ImShape[1]
    return np.array([ImW * 0.5, 0.25 * ImH]), ImW


def YTracking_Objective_Function(image, color):
    # variance method
    if image.size != 0:
        if color:
            image = bgr2gray(image)
        mean, std = cv2.meanStdDev(image)
        return std[0][0] ** 2
    else:
        return 0


def calculate_focus_measure(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # optional
    lap = cv2.Laplacian(image, cv2.CV_16S)
    focus_measure = mean(square(lap))
    return focus_measure


# test part
if __name__ == "__main__":
    # Load an color image in grayscale
    rouge = np.array([[[255, 0, 0]]], dtype="uint8")
    vert = np.array([[[0, 255, 0]]], dtype="uint8")
    bleu = np.array([[[0, 0, 255]]], dtype="uint8")

    rouge_HSV = cv2.cvtColor(rouge, cv2.COLOR_RGB2HSV)[0][0]
    vert_HSV = cv2.cvtColor(vert, cv2.COLOR_RGB2HSV)[0][0]
    bleu_HSV = cv2.cvtColor(bleu, cv2.COLOR_RGB2HSV)[0][0]

    img = cv2.imread(
        "C:/Users/Francois/Documents/11-Stage_3A/6-Code_Python/ConsoleWheel/test/rouge.jpg"
    )
    print(img)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    couleur = bleu_HSV
    LOWER = default_lower_HSV(couleur)
    UPPER = default_upper_HSV(couleur)

    img3 = threshold_image(img2, LOWER, UPPER)
    cv2.imshow("image", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# for more than one tracked object
"""
def find_centroid_many(image,contour_area_min,contour_area_max):
    contours = cv2.findContours(image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    count=0
    last_centroids=[]
    for j in range(len(contours)):
        cnt = contours[j]
        if cv2.contourArea(contours[j])>contour_area_min and cv2.contourArea(contours[j])<contour_area_max :
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            last_centroids.append([cx,cy])
            count+=1
    return last_centroids,count
"""
import cv2
import numpy as np
from numpy import mean, square, std
from scipy.ndimage import label


def crop_image(image,crop_width,crop_height):
    image_height = image.shape[0]
    image_width = image.shape[1]
    roi_left = int(max(image_width/2 - crop_width/2,0))
    roi_right = int(min(image_width/2 + crop_width/2,image_width))
    roi_top = int(max(image_height/2 - crop_height/2,0))
    roi_bottom = int(min(image_height/2 + crop_height/2,image_height))
    image_cropped = image[roi_top:roi_bottom,roi_left:roi_right]
    return image_cropped

def calculate_focus_measure(image,method='LAPE'):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # optional
    if method == 'LAPE':
        if image.dtype == np.uint16:
            lap = cv2.Laplacian(image,cv2.CV_32F)
        else:
            lap = cv2.Laplacian(image,cv2.CV_16S)
        focus_measure = mean(square(lap))
    elif method == 'GLVA':
        focus_measure = np.std(image,axis=None)# GLVA
    else:
        focus_measure = np.std(image,axis=None)# GLVA
    return focus_measure

def unsigned_to_signed(unsigned_array,N):
    signed = 0
    for i in range(N):
        signed = signed + int(unsigned_array[i])*(256**(N-1-i))
    signed = signed - (256**N)/2
    return signed

def rotate_and_flip_image(image,rotate_image_angle,flip_image):
    ret_image = image.copy()
    if(rotate_image_angle != 0):
        '''
            # ROTATE_90_CLOCKWISE
            # ROTATE_90_COUNTERCLOCKWISE
        '''
        if(rotate_image_angle == 90):
            ret_image = cv2.rotate(ret_image,cv2.ROTATE_90_CLOCKWISE)
        elif(rotate_image_angle == -90):
            ret_image = cv2.rotate(ret_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif(rotate_image_angle == 180):
            ret_image = cv2.rotate(ret_image,cv2.ROTATE_180)

    if(flip_image is not None):
        '''
            flipcode = 0: flip vertically
            flipcode > 0: flip horizontally
            flipcode < 0: flip vertically and horizontally
        '''
        if(flip_image == 'Vertical'):
            ret_image = cv2.flip(ret_image, 0)
        elif(flip_image == 'Horizontal'):
            ret_image = cv2.flip(ret_image, 1)
        elif(flip_image == 'Both'):
            ret_image = cv2.flip(ret_image, -1)

    return ret_image

def generate_dpc(im_left, im_right):
    # Normalize the images
    im_left = im_left.astype(float)/255
    im_right = im_right.astype(float)/255
    # differential phase contrast calculation
    im_dpc = 0.5 + np.divide(im_left-im_right, im_left+im_right)
    # take care of errors
    im_dpc[im_dpc < 0] = 0
    im_dpc[im_dpc > 1] = 1
    im_dpc[np.isnan(im_dpc)] = 0

    im_dpc = (im_dpc * 255).astype(np.uint8)

    return im_dpc

def colorize_mask(mask):
    # Label the detected objects
    labeled_mask, ___ = label(mask)
    # Color them
    colored_mask = np.array((labeled_mask * 83) % 255, dtype=np.uint8)
    colored_mask = cv2.applyColorMap(colored_mask, cv2.COLORMAP_HSV)
    # make sure background is black
    colored_mask[labeled_mask == 0] = 0
    return colored_mask

def colorize_mask_get_counts(mask):
    # Label the detected objects
    labeled_mask, no_cells = label(mask)
    # Color them
    colored_mask = np.array((labeled_mask * 83) % 255, dtype=np.uint8)
    colored_mask = cv2.applyColorMap(colored_mask, cv2.COLORMAP_HSV)
    # make sure background is black
    colored_mask[labeled_mask == 0] = 0
    return colored_mask, no_cells

def overlay_mask_dpc(color_mask, im_dpc):
    # Overlay the colored mask and DPC image
    # make DPC 3-channel
    im_dpc = np.stack([im_dpc]*3, axis=2)
    return (0.75*im_dpc + 0.25*color_mask).astype(np.uint8)

def centerCrop(image, crop_sz):
    center = image.shape
    x = int(center[1]/2 - crop_sz/2)
    y = int(center[0]/2 - crop_sz/2)
    cropped = image[y:y+crop_sz, x:x+crop_sz]

    return cropped

def interpolate_plane(triple1, triple2, triple3, point):
    """
    Given 3 triples triple1-3 of coordinates (x,y,z)
    and a pair of coordinates (x,y), linearly interpolates
    the z-value at (x,y).
    """
    # Unpack points
    x1, y1, z1 = triple1
    x2, y2, z2 = triple2
    x3, y3, z3 = triple3

    x,y = point
    # Calculate barycentric coordinates
    detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if detT == 0:
        raise ValueError("Your 3 x-y coordinates are linear")
    alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / detT
    beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / detT
    gamma = 1 - alpha - beta

    # Interpolate z-coordinate
    z = alpha * z1 + beta * z2 + gamma * z3

    return z

