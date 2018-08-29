import os
import numpy as np
import cv2
from shape import Shape, ShapeList
from models import PointDistributionModel,AppearanceModel

def appearance_model_eight_teeth(training_images,training_landmarks,test_image,pyr_levels=0,top_extent=[6,6]):
    scaled_training_images,scaled_training_landmarks = gaussian_pyramid_down(training_images,num_levels=pyr_levels, training_landmarks=training_landmarks)
    scaled_test_image = gaussian_pyramid_down([test_image],num_levels=pyr_levels)[0]
    shape_model = PointDistributionModel(scaled_training_landmarks)
    app_model = AppearanceModel(scaled_training_images,shape_model,top_extent,[1,1])
    initial_shape = app_model.fit(scaled_test_image)
    for i in range(1,pyr_levels):
        initial_shape=initial_shape.pyr_up()
    return initial_shape

def appearance_model_four_teeth(training_images,training_landmarks,test_image,pyr_levels=0,top_extent=[7,7],top_scale=[1.1,1.1],bottom_extent=[6,6],bottom_scale=[1.05,1.1]):
    scaled_training_images,scaled_training_landmarks = gaussian_pyramid_down(training_images,num_levels=pyr_levels, training_landmarks=training_landmarks)
    scaled_test_image = gaussian_pyramid_down([test_image],num_levels=pyr_levels)[0]
    transformed_test_image = scaled_test_image
    transformed_training_images =  scaled_training_images
    split_training_landmarks = tooth_splitter(scaled_training_landmarks, 2)
    shape_model_top = PointDistributionModel(split_training_landmarks[0])
    shape_model_bottom = PointDistributionModel(split_training_landmarks[1])
    app_model_top = AppearanceModel(transformed_training_images,shape_model_top,top_extent,top_scale)
    app_model_bottom = AppearanceModel(transformed_training_images,shape_model_bottom,bottom_extent,bottom_scale)
    shape_top= app_model_top.fit(transformed_test_image)
    shape_bottom= app_model_bottom.fit(transformed_test_image)
    shape = shape_top.concatenate(shape_bottom)
    for i in range(1,pyr_levels):
        shape=shape.pyr_up()
    return shape

def tooth_models(all_training_landmarks, all_test_landmark, pca_variance_captured=[0.9,0.9,0.9,0.9],
                 project_to_tangent_space=False):
    model_list = []
    error_list = []
    training_landmarks_list = []
    test_landmark_list = []
    ctr = 0
    for num_parts in [1, 2, 4, 8]:
        split_training_landmarks = tooth_splitter(all_training_landmarks, num_parts)
        split_test_landmark = tooth_splitter([all_test_landmark], num_parts)
        sub_model_list = []
        sub_error_list = []
        sub_training_landmarks_list = []
        sub_test_landmark_list = []
        for split_part in range(num_parts):
            training_landmarks = split_training_landmarks[split_part]
            test_landmark = split_test_landmark[split_part][0]
            model = PointDistributionModel(training_landmarks, project_to_tangent_space=project_to_tangent_space,
                                           pca_variance_captured=pca_variance_captured[ctr])
            _, fit_error, _ = model.fit(test_landmark)
            sub_error_list.append(fit_error / test_landmark.get_size())
            sub_model_list.append(model)
            sub_test_landmark_list.append(test_landmark)
            sub_training_landmarks_list.append(training_landmarks)
        model_list.append(sub_model_list)
        error_list.append(sub_error_list)
        training_landmarks_list.append(sub_training_landmarks_list)
        test_landmark_list.append(sub_test_landmark_list)
        ctr+=1

    return model_list, training_landmarks_list, test_landmark_list, error_list


def tooth_splitter(complete_landmarks, num_parts):
    return [ShapeList(list(item)) for item in
            list(zip(*[tuple(ShapeList.from_shape(shape, num_parts)) for shape in complete_landmarks]))]


def gaussian_pyramid_down(training_images, num_levels, training_landmarks=None, complete=False):
    p_training_images = [training_images]
    p_training_landmarks = []
    if training_landmarks is not None:
        p_training_landmarks = [training_landmarks]
    for l in range(1, num_levels):
        p_training_images.append([cv2.pyrDown(image) for image in p_training_images[l - 1]])
        if training_landmarks is not None:
            p_training_landmarks.append(ShapeList([shape.pyr_down() for shape in p_training_landmarks[l - 1]]))
    if complete or (training_images is None and training_landmarks is None):
        return p_training_images, p_training_landmarks
    if training_landmarks is None:
        if not complete:
            return p_training_images[num_levels - 1]
        else:
            return p_training_images
    elif training_landmarks is not None:
        if not complete:
            return p_training_images[num_levels - 1], p_training_landmarks[num_levels - 1]
        else:
            return p_training_images, p_training_landmarks


def gaussian_pyramid_up(training_images, num_levels, training_landmarks=None, complete=False):
    p_training_images = [training_images]
    p_training_landmarks = []
    if training_landmarks is not None:
        p_training_landmarks = [training_landmarks]
    for l in range(1, num_levels):
        p_training_images.append([cv2.pyrUp(image) for image in p_training_images[l - 1]])
        if training_landmarks is not None:
            p_training_landmarks.append(ShapeList([shape.pyr_up() for shape in p_training_landmarks[l - 1]]))
    if complete or (training_images is None and training_landmarks is None):
        return p_training_images, p_training_landmarks
    if training_landmarks is None:
        if not complete:
            return p_training_images[num_levels - 1]
        else:
            return p_training_images
    elif training_landmarks is not None:
        if not complete:
            return p_training_images[num_levels - 1], p_training_landmarks[num_levels - 1]
        else:
            return p_training_images, p_training_landmarks


def load_landmark(filepath, mirrored=False, width=0):
    """
    Creates a shape from a file containing 2D points
    in the following format
        x1
        y1
        x2
        y2
        ...
        xn
        yn
    :param filepath: The path to the landmark file
    :param mirrored: True when reading a vertically mirrored landmark
    :param width: The image width, needed when reading a mirrored landmark
    :return: A Shape object
    """
    y_list = []
    x_list = []
    if mirrored and width == 0:
        raise ValueError("Need a nonzero width for a mirrored landmark")
    with open(filepath) as fd:
        for i, line in enumerate(fd):
            if i % 2 == 0:
                x_list.append(float(line) + width)
            else:
                y_list.append(float(line))
    return Shape.from_coordinate_lists(x_list, y_list)


def load_image(filepath):
    """
    Creates a 2D Array from a file containing the
    image/segmentation
    :param filepath:  The path to the image/segmentation
    :return: A 2D Numpy Array
    """
    return cv2.imread(filepath, 0)


def parse_segmentation(img):
    """
    Creates a list of pixels from a 2D Array containing
    the segmentation
    :param img: The img as a 2D Array
    :return: A list of coordinates indices for non zero pixels
    """
    img2 = np.uint8(img.copy())
    img2[img2 > 0] = 1
    return img2




class Dataset:
    """
        Class to represent the data for the assignment
    """
    _training_image_count = 14
    _test_image_count = 16
    _tooth_count = 8
    TOP_TEETH = range(4, 8)
    BOTTOM_TEETH = range(4)
    MIDDLE_TEETH = [1, 2, 5, 6]
    ALL_TEETH = range(_tooth_count)
    ALL_TRAINING_IMAGES = range(_training_image_count)
    
    def __init__(self, data_folder):
        self._training_images = []
        self._training_images_mirrored = []
        self._training_landmarks = []
        self._training_landmarks_mirrored = []
        self._training_segmentations = []
        self._training_segmentations_mirrored = []
        self._data_folder = data_folder
        self._read_training_data()
        self._read_extra_images()
        
    def _read_training_data(self):
        for image_index in range(self._training_image_count):
            original_image, mirrored_image, width = self._process_radiograph(image_index)
            self._training_images.append(original_image)
            self._training_images_mirrored.append(mirrored_image)
            landmarks = []
            segmentations = []
            landmarks_mirrored = []
            segmentations_mirrored = []
            for tooth_index in range(self._tooth_count):
                original_landmark, mirrored_landmark = self._process_tooth_landmarks(image_index, tooth_index, width)
                landmarks.append(original_landmark)
                landmarks_mirrored.append(mirrored_landmark)
                original_segmentation, mirrored_segmentation = self._process_tooth_segmentations(image_index,
                                                                                                 tooth_index)
                segmentations.append(original_segmentation)
                segmentations_mirrored.append(mirrored_segmentation)
            self._training_landmarks.append(landmarks)
            self._training_landmarks_mirrored.append(landmarks_mirrored)
            self._training_segmentations.append(segmentations)
            self._training_segmentations_mirrored.append(segmentations_mirrored)
            
    def _process_radiograph(self, image_index):
        original_image = load_image(self._build_image_filepath(image_index))
        _, width = original_image.shape
        mirrored_image = cv2.flip(original_image, 1)
        return original_image, mirrored_image, width

    def _build_image_filepath(self, image_index):
        radiograph_filepath_prefix = os.path.join(self._data_folder, 'Radiographs')
        return os.path.join(radiograph_filepath_prefix, str(image_index + 1).zfill(2) + '.tif')
    
    def _process_tooth_landmarks(self, image_index, tooth_index, width):
        original_landmark = load_landmark(self._build_landmark_filepath(image_index, tooth_index))
        mirrored_landmark = load_landmark(self._build_landmark_filepath(image_index, tooth_index, True), True, width)
        return original_landmark, mirrored_landmark

    def _build_landmark_filepath(self, image_index, tooth_index, mirrored=False):
        landmarks_filepath_prefix = os.path.join(self._data_folder, 'Landmarks')
        if mirrored:
            landmarks_filepath_prefix = os.path.join(landmarks_filepath_prefix, 'mirrored')
            image_index += self._training_image_count
        else:
            landmarks_filepath_prefix = os.path.join(landmarks_filepath_prefix, 'original')
        return os.path.join(landmarks_filepath_prefix,
                            'landmarks' + str(image_index + 1) + '-' + str(tooth_index + 1) + '.txt')

    def _build_segmentation_filepath(self, image_index, tooth_index):
        segmentations_filepath_prefix = os.path.join(self._data_folder, 'Segmentations')
        return os.path.join(segmentations_filepath_prefix,
                            str(image_index + 1).zfill(2) + '-' + str(tooth_index) + '.png')


    def _build_extra_image_filepath(self, image_index):
        radiograph_filepath_prefix = os.path.join(self._data_folder, os.path.join('Radiographs', 'extra'))
        return os.path.join(radiograph_filepath_prefix, str(image_index + 1).zfill(2) + '.tif')


    def _process_tooth_segmentations(self, image_index, tooth_index):
        segmentation_img = load_image(self._build_segmentation_filepath(image_index, tooth_index))
        original_segmentation = parse_segmentation(segmentation_img)
        mirrored_segmentation = parse_segmentation(cv2.flip(segmentation_img, 1))
        return original_segmentation, mirrored_segmentation

    def _process_radiograph(self, image_index):
        original_image = load_image(self._build_image_filepath(image_index))
        _, width = original_image.shape
        mirrored_image = cv2.flip(original_image, 1)
        return original_image, mirrored_image, width

    def _read_extra_images(self):
        self._extra_images = []
        for image_index in range(self._test_image_count):
            self._extra_images.append(
                load_image(self._build_extra_image_filepath(image_index + self._training_image_count)))

    

    def get_training_images(self, image_indices):
        images = []
        mirrored_images = []
        for image_index in image_indices:
            mirrored_images.append(self._training_images_mirrored[image_index])
            images.append(self._training_images[image_index])
        return images, mirrored_images

    def get_extra_images(self, image_indices):
        images = []
        for image_index in image_indices:
            images.append(self._extra_images[image_index])
        return images

    def get_training_image_segmentations(self, image_indices, tooth_indices):
        segmentations = []
        mirrored_segmentations = []
        for image_index in image_indices:
            combined_segmentation = np.uint8(np.zeros(self._training_segmentations[image_index][0].shape))
            combined_segmentation_mirrored = np.uint8(np.zeros(self._training_segmentations[image_index][0].shape))
            for tooth_index in tooth_indices:
                combined_segmentation = np.bitwise_or(combined_segmentation,
                                                      self._training_segmentations[image_index][tooth_index])
                combined_segmentation_mirrored = np.bitwise_or(combined_segmentation_mirrored,
                                                               self._training_segmentations_mirrored[image_index][
                                                                   tooth_index])
            segmentations.append(combined_segmentation)
            mirrored_segmentations.append(combined_segmentation_mirrored)
        return segmentations, mirrored_segmentations

   
    def get_training_image_landmarks(self, image_indices, tooth_indices):
        """
        This returns the landmarks for the given image and teeth indices
        :param image_indices: A list containing the image indices for which the landmarks must be fetched
        :param tooth_indices: A list containing the tooth indices for which the landmarks must be fetched
            e.g TOP_TEETH is [4,5,6,7], BOTTOM_TEETH is [0,1,2,3]
        :return: Two ShapeLists - containing the mirrored and unmirrored landmarks
        """
        landmarks = []
        mirrored_landmarks = []
        for image_index in image_indices:
            final_landmark = None
            final_landmark_mirrored = None
            for tooth_index in tooth_indices:
                landmark = self._training_landmarks[image_index][tooth_index]
                mirrored_landmark = self._training_landmarks_mirrored[image_index][tooth_index]
                if final_landmark is None:
                    final_landmark = landmark
                    final_landmark_mirrored = mirrored_landmark
                else:
                    final_landmark = final_landmark.concatenate(landmark)
                    final_landmark_mirrored = final_landmark_mirrored.concatenate(mirrored_landmark)
            landmarks.append(final_landmark)
            mirrored_landmarks.append(final_landmark_mirrored)
        return ShapeList(landmarks), ShapeList(mirrored_landmarks)

    


class LeaveOneOutSplitter:
    def __init__(self, data, images_indices=Dataset.ALL_TRAINING_IMAGES, shapes_indices=Dataset.ALL_TEETH):
        img, mimg = data.get_training_images(images_indices)
        l, ml = data.get_training_image_landmarks(images_indices, shapes_indices)
        s, ms = data.get_training_image_segmentations(images_indices, shapes_indices)
        images = img + mimg
        shapes = l.concatenate(ml)
        segmentations = s + ms
        self._images = images
        self._shapes = shapes
        self._segmentations = segmentations
        self._test_idx = -1
        self._training_idx = []

    def get_training_set_size(self):
        return len(self._training_idx)


    def get_test_index(self):
        return self._test_idx

    def get_training_set(self):
        return [self._images[idx] for idx in self._training_idx], ShapeList(
            [self._shapes[idx] for idx in self._training_idx]), [
                   self._segmentations[idx] for idx in self._training_idx]

    def get_test_example(self):
        return self._images[self._test_idx], self._shapes[self._test_idx], self._segmentations[self._test_idx]

    def get_dice_error_on_test(self, detected_shape):
        shape = detected_shape.round()
        bin_truth = self._segmentations[self._test_idx]
        bin_predicted = np.uint8(np.zeros(bin_truth.shape))
        cv2.drawContours(bin_predicted, ShapeList.from_shape(detected_shape, 8).as_list_of_contours(), -1,
                         (255, 255, 255), -1)
        bin_predicted[bin_predicted > 0] = 1
        intersection = float(np.sum(np.sum(np.bitwise_and(bin_truth, bin_predicted))))
        union = float(np.sum(np.sum(np.bitwise_or(bin_truth, bin_predicted))))
        return intersection / union

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def next(self):
        if self._test_idx > len(self._images) / 2 - 1:
            raise StopIteration
        else:
            self._test_idx += 1
            
            self._training_idx = list(range(0, self._test_idx)) + list(range(self._test_idx + 1, len(self._images) // 2)) + list(range(
                len(self._images) // 2, self._test_idx + len(self._images) // 2)) + list(range(
                self._test_idx + 1 + len(self._images) // 2, len(self._images)))
            return self
