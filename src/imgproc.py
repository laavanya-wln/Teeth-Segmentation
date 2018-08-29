import numpy as np
import cv2
from shape import Shape, ShapeList, LineGenerator

__author__ = "Laavanya Wudali Lakshmi Narsu"

"""
Methods for image processing
"""
def patch_transformation_default(data):
    h,w = data.shape
    result = cv2.Sobel(data,6,0,1,ksize=5)
    result = cv2.medianBlur(cv2.convertScaleAbs(result),3)
    return result

def image_transformation_default(img):
    return img



"""
For every training image and its landmarks,

"""
def extract_patch_normal(orig_img,shape,num_pixels_length, num_pixels_width,normal_point_neighborhood=2,image_transformation_function=image_transformation_default,patch_transformation_function=patch_transformation_default):
    if image_transformation_function is None:
        img=orig_img
    else:
        img = image_transformation_function(orig_img)
    h,w=img.shape 
    all_patches = []
    all_points = []
    for point_index in range(shape.get_size()):
        point = shape.get_point(point_index)
        tangent_slope_vector, normal_slope_vector = shape.get_slope_vectors_at_point(point_index,normal_point_neighborhood)
        normal_coordinates_generator = LineGenerator(point, normal_slope_vector)
        normal_coordinate_list = normal_coordinates_generator.generate_two_sided(num_pixels_length)
        all_points.append(normal_coordinate_list)
        all_pixels = []
        for coordinates in normal_coordinate_list:
            tangent_coordinates_generator = LineGenerator(coordinates, tangent_slope_vector)
            tangent_coordinate_list=tangent_coordinates_generator.generate_two_sided(num_pixels_width)
            row_pixels = []
            for coordinates in tangent_coordinate_list:
                if 0 <= coordinates[1] < h and 0 < coordinates[0] < w:
                    row_pixels.append(img[coordinates[1], coordinates[0]])
                else:
                    raise ValueError("Index exceeds image dimensions")
            all_pixels.append(row_pixels)
        patch_data  = np.array(all_pixels)
        if patch_transformation_function is not None:
            patch_data = patch_transformation_function(patch_data)
        all_patches.append(patch_data)
    return all_patches,np.array(all_points)