import sys
import numpy as np
import scipy.io as sio
import cv2


def create_normal(d_im):
    d_im = d_im.astype("float64")
    data_type = "float64"
    normals = np.zeros((d_im.shape[0],d_im.shape[1], 2), dtype=data_type)
    h , w, o = normals.shape
    for i in range(1,w-1):
        for j in range(1, h-1):
            v1 = np.array([d_im[j,i], d_im[j-1,i]], dtype=data_type)
            v2 = np.array([d_im[j,i], d_im[j,i-1]], dtype=data_type)
            d = np.cross(v1, v2)
            if d != 0:
                n = d / np.sqrt((np.sum(d**2)))
            else:
                n = d
            normals[j,i,:] = n   # changed n to d
    return normals

def angle_to_color(angle):
    M_PI = np.pi
    red   =  np.sin(angle)
    green =  np.sin(angle + 2*M_PI / 3.)
    blue  =   np.sin(angle + 4*M_PI / 3.)
    return red, green, blue

def get_angle(real, imag):
    z = complex(real, imag)
    angle = np.angle(z, deg=True)
    if angle < 0:
        angle += 360
    return angle

def create_surface_normal(d_im):
    """
    0 is black 255 white
    """
    data_type = "float64"
    color_normal = np.zeros((d_im.shape[0],d_im.shape[1],3), dtype=data_type)
    h , w, d = d_im.shape
    for i in range(w):
        for j in range(h):
            angle = get_angle(d_im[j,i, 0], d_im[j,i, 1])
            r,g, b = angle_to_color(angle)
            color_normal[j,i, 0] = r
            color_normal[j,i, 1] = g
            color_normal[j,i, 2] = b

    return color_normal


d1 = cv2.imread("depth_kitchen_d1.png", 0)
d1_n = (d1 - d1.min()) / (d1.max() - d1.min())
d1_n_255 = d1_n * 255
depth = d1_n_255.astype(np.uint8)
cv2.imshow("depth", depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(depth.shape)
res = create_normal(depth)
color_norm = create_surface_normal(res)
cv2.imshow("la", color_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("normal_kitchen_d1.png", color_norm)



