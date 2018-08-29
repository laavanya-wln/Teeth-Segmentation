import numpy as np
import cv2

__author__ = "Laavanya Wudali Lakshmi Narsu"

"""
Data structures to hold shapes and shape lists
"""


class Shape:
    """
    Represents a 2D shape

    Attributes:
        _data The points stored as a nxd numpy matrix

    Authors: Laavanya Wudali Lakshmi Narsu
    """

    """
    CONSTRUCTORS
    """

    def __init__(self, data):
        """
        Initializes a shape from a nx2 numpy matrix of points
        :param data: a numpy matrix of shape (n,2)
        """
        if data is not None:
            n, d = data.shape
            if d != 2:
                raise ValueError("Class currently only supports 2D shapes")
            self._data = np.copy(data)
        else:
            self._data = np.array([])

    @classmethod
    def from_collapsed_vector(cls, shape_vector):
        """
        Creates a new shape from a numpy 1D array of the form [x1 y1 x2 y2 x3 y3 ...]
        :param shape_vector: a 1D numpy array
        :return: A shape
        """
        return cls(np.reshape(shape_vector, (len(shape_vector.tolist()) // 2, 2)))

    @classmethod
    def from_coordinate_lists(cls, x_list, y_list):
        """
        Creates a shape from two same-sized coordinate lists
        :param x_list: The list of x-coordinated
        :param y_list: The list of y-coordinates of
        the same length as x_list
        :return: A Shape object
        """
        if len(x_list) != len(y_list):
            raise AssertionError("The number of x and y coordinates are different")
        points = []
        for i in range(len(x_list)):
            points.append([x_list[i], y_list[i]])
        return Shape(np.array(points))
    
    @classmethod
    def from_contour(cls,contour):
        points = []        
        for wierdness in contour:
            points.append(wierdness[0])
        return Shape(np.array(points))

    '''
    VIEWS
    '''

    def as_numpy_matrix(self):
        """
        Returns the points as a nx2 numpy matrix
        NOTE: A separate copy is returned
        :return: a numpy matrix of shape (n,2)
        """
        return np.copy(self._data)

    def as_list_of_points(self):
        """
        Returns the data as a list of lists
        NOTE: A separate copy is returned
        :return: A list of list of points
        """
        return self._data.tolist()

    def as_contour(self):
        """
        Returns the shape as a contour usable by cv
        :return: a cv contour as a 2d numpy matrix
        """
        contour_list = []
        for point in self._data.tolist():
            contour_list.append([point])
        return np.int32(np.array(contour_list))

    def as_collapsed_vector(self):
        """
        Collapses the shape into a vector
        :return: A vector of 2*size() points
        """
        n, _ = self._data.shape
        return np.reshape(self._data, 2 * n)

    '''
    SHAPE ATTRIBUTES
    '''

    def get_size(self):
        """
        Returns the number of points in the shape
        :return: The number of points
        """
        r, _ = self._data.shape
        return r

    def get_bounding_box(self):
        raw_data = self.as_numpy_matrix()
        return np.array([np.min(raw_data, axis=0), np.max(raw_data, axis=0)])

    def get_centroid(self):
        """
        Returns the centroid point of the shape points
        :return: A 2d numpy array
        """
        return np.mean(self._data, axis=0)

    def get_norm(self):
        """
        Returns the column norms of the shape points as a vector of [x_norm y_norm]
        :return: A 2d numpy array containing the column norms
        """
        return np.linalg.norm(self._data, axis=0)

    def get_point(self, point_index):
        """
        Returns the point at the given index
        :param point_index:  The index of the query point in the shape
        :return: A 2D array containing the point coordinated
        """
        if 0 <= point_index < self.get_size():
            return self._data[point_index]
        return np.array([])

    def get_sub_shape(self, point_indices):
        """
        Returns a sub shape consisting of the selected points
        :param point_indices: A list of point indices
        :return: A shape object
        """
        return [self.get_point(point_index) for point_index in point_indices]
        
    def get_convex_hull(self):
        return Shape.from_contour(cv2.convexHull(self.as_contour(),returnPoints=True))

    def _get_neighborhood(self, point_index, num_neighbors):
        """
        Returns the neighborhood around a given point.
        :param point_index: The index of the query point in the shape
        :param num_neighbors: The number of neighborhood points needed ON EITHER SIDE
        :return: An array of neighborhood points
        """
        if 0 <= point_index < self.get_size():
            neighborhood_index_increments = list(range(-num_neighbors, 0, 1)) + list(range(1, num_neighbors + 1, 1))
            neighborhood_indices = [(point_index + increment) % self.get_size() for increment in
                                    neighborhood_index_increments]
            return np.array([self._data[index] for index in neighborhood_indices])
        return np.array([])

    def get_slope_vectors_at_point(self, point_index, normal_neighborhood):
        neighborhood = self._get_neighborhood(point_index, normal_neighborhood)
        line = cv2.fitLine(np.int32(neighborhood), 2, 0, 0.01, 0.01)
        tangent_slope_vector = [line[0], line[1]]
        normal_slope_vector = [-tangent_slope_vector[1], tangent_slope_vector[0]]
        return np.array(tangent_slope_vector), np.array(normal_slope_vector)

    def get_transformation(self, other, w=None):#, allow_affine=False,):
        """
        Returns the transformation matrix (homogeneous coordinates)
        to transform this shape to the other shape
        Refer Appendix C of Cootes, Timothy F., and Cristopher J. Taylor.
        "Statistical models of appearance for computer vision." (2004).
        for more details
        :param other: The other Shape
        :return: The 3x3 transformation matrix
        """
        if w is None:
            n = self.get_size()
            p = self.as_numpy_matrix()
            q = other.as_numpy_matrix()
            sx = np.sum(p[:, 0])
            sy = np.sum(p[:, 1])
            sxd = np.sum(q[:, 0])
            syd = np.sum(q[:, 1])
            sxx = np.sum(np.power(p[:, 0], 2))
            sxy = np.sum(np.multiply(p[:, 0], p[:, 1]))
            syy = np.sum(np.power(p[:, 1], 2))
            sxxd = np.sum(np.multiply(p[:, 0], q[:, 0]))
            syyd = np.sum(np.multiply(p[:, 1], q[:, 1]))
            sxyd = np.sum(np.multiply(p[:, 0], q[:, 1]))
            syxd = np.sum(np.multiply(q[:, 1], p[:, 0]))
            lhs = np.array([[sxx + syy, 0, sx, sy],
                                [0, sxx + syy, -sy, sx],
                                [sx, -sy, n, 0],
                                [sy, sx, 0, n]])
            rhs = np.array([sxxd + syyd, sxyd - syxd, sxd, syd])
            tmat = np.dot(np.linalg.pinv(lhs), rhs)
            a, b, tx, ty = tuple(tmat.tolist())
            return np.array([[a, b, 0], [-b, a, 0], [tx, ty, 1]])
            #if allow_affine:
            #    lhs = np.array([[sxx, sxy, sx],
            #                    [sxy, syy, sy],
            #                    [sx, sy, n]])
            #    rhs = np.array([[sxxd, sxyd], [syxd, syyd], [sxd, syd]])
            #else:
            #    lhs = np.array([[sxx + syy, 0, sx, sy],
            #                    [0, sxx + syy, -sy, sx],
            #                    [sx, -sy, n, 0],
            #                    [sy, sx, 0, n]])
            #    rhs = np.array([sxxd + syyd, sxyd - syxd, sxd, syd])
            
            #if allow_affine:
            #    a = tmat[0, 0]
            #    b = tmat[1, 0]
            #    c = tmat[0, 1]
            #    d = tmat[1, 1]
            #    tx = tmat[2, 0]
            #    ty = tmat[2, 1]
            #    return np.array([[a, c, 0], [b, d, 0], [tx, ty, 1]])
            #else:
            #    a, b, tx, ty = tuple(tmat.tolist())
            #    return np.array([[a, b, 0], [-b, a, 0], [tx, ty, 1]])
        else:
            W = np.sum(w)
            s1 = self.as_numpy_matrix()
            s2 = other.as_numpy_matrix()
            x1 = s1[:,0]
            y1 = s1[:,1]
            x2 = s2[:,0]
            y2 = s2[:,1]
            X1 = np.dot(w,x1)
            X2 = np.dot(w,x2)
            Y1 = np.dot(w,y1)
            Y2 = np.dot(w,y2)
            Z = np.dot(w,x2**2+y2**2)    
            C1 = np.dot(w,np.multiply(x1,x2)+np.multiply(y1,y2))
            C2 = np.dot(w,np.multiply(y1,x2)-np.multiply(x1,y2))
            A = np.array([[X2,-Y2,W,0],[Y2,X2,0,W],[Z,0,X2,Y2],[0,Z,-Y2,X2]])
            B = np.array([X1,Y1,C1,C2])
            R=np.linalg.solve(A, B)
            return np.array([[R[0],-R[1],0],[R[1],R[0],0],[R[2],R[3],1]])

    '''
    SHAPE TRANSFORMATIONS AND OPERATIONS
    '''

    def concatenate(self, other):
        """
        Returns a new shape combining the points in other and the current shape
        :param other: The  other shape
        :return: A shape containing all the points in the current and other shapes
        """
        return Shape(np.concatenate((self._data, other.as_numpy_matrix())))

    def center(self):
        """
        Translates the shape such that
        the mean is at the origin
        :return: A Shape object, with mean as
        zero vector
        """
        return Shape(self._data - self.get_centroid())

    def normalize(self, normalize_over_all_elements=True):
        """
        Returns a new shape containing this shape
        scaled to unit norm
        :param normalize_over_all_elements: |x|=1
        :return: A Shape object, scaled to unit norm
        """
        if normalize_over_all_elements:
            return Shape(self._data / np.linalg.norm(self.as_collapsed_vector()))
        return Shape(self._data / self.get_norm())

    def project_to_tangent_space(self, other):
        """
        Projects the current space to the tangent space of the other shape
        :param other: The other Shape
        :return: The tangent projected shape
        """
        return Shape(self._data / np.dot(other.as_collapsed_vector(), self.as_collapsed_vector()))

    def scale(self, scaling):
        """
        Returns the current shape scaled by the specified displacement vector
        :param scaling: The 2d vector of scaling
        :return: A shape object containing the scaled shape
        """
        return Shape(self._data * scaling)

    def translate(self, displacement):
        """
        Returns the current shape translated by the specified displacement vector
        :param displacement: The 2d vector of displacements
        :return: A shape object containing the displaced shape
        """
        return Shape(self._data + displacement)

    def transform(self, hmat):
        self_data = np.concatenate((self.as_numpy_matrix(), np.ones((self.get_size(), 1), dtype=float)), axis=1)
        return Shape(np.dot(self_data, hmat)[:, 0:2])

    def align(self, other, use_transformation_matrix=True,weights=None):
        """
        Aligns the current shape
        to the other shape  by
        finding a transformation matrix  a=sr by solving the
        least squares solution of the equation
        self*a= other
        :param use_transformation_matrix: Use get_transformation_matrix()
        :param other: The other shape
        :return: A shape aligned to other
        """
        if use_transformation_matrix:
            return self.transform(self.get_transformation(other,w=weights))
        translation = other.get_centroid() - self.get_centroid()
        other_data = other.as_numpy_matrix() - other.get_centroid()
        self_data = self._data - self.get_centroid()
        cov = np.dot(other_data.T, self_data)
        btb = np.dot(other_data.T, other_data)
        pic = np.linalg.pinv(cov)
        a = np.dot(pic, btb)
        return Shape(np.dot(self_data, a) + translation)

    def round(self):
        """
        Returns a new shape with coordinates rounded to the nearest integer
        :return:
        """
        return Shape(np.int32(np.round(self.as_numpy_matrix())))

    def pyr_down(self):
        """
        Returns a new shape with the coordinates scaled down by 2
        :return: A shape object
        """
        return Shape(self.as_numpy_matrix() / 2)

    def pyr_up(self):
        """
        Returns a new shape with the coordinates scaled up by 2
        :return: A shape object
        """
        return Shape(np.round(self.as_numpy_matrix() * 2))


class ShapeList:
    """
    A wrapper class for a list of Shapes - behaves like a list

    Attributes:
        _shapes a numpy matrix of shapes

    Authors: Laavanya Wudali Lakshmi Narsu
    """

    def __init__(self, shapes):
        self._shapes = shapes

    def __iter__(self):
        return self._shapes.__iter__()

    def __len__(self):
        return len(self._shapes)

    def __getitem__(self, index):
        return self._shapes[index]

    @classmethod
    def from_shape(cls, shape, num_parts):
        """
        Splits the shape into a ShapeList of num_parts parts
        :param shape: The shape
        :param num_parts: Number of parts
        :return: ShapeList
        """
        return cls([Shape(part) for part in np.split(shape.as_numpy_matrix(), num_parts)])

    def as_numpy_matrix(self):
        """
        Returns the shape list as a number of shapes x number of points x 2 array
        :return: A numpy matrix
        """
        return np.array([shape.as_numpy_matrix() for shape in self._shapes])

    def as_list_of_shapes(self):
        """
        Returns the shapes as a list of shape objects
        :return: A list of shape objects
        """
        return self._shapes

    def as_list_of_contours(self):
        """
        Returns the shapes as a list of contours
        :return: A list of shape objects
        """
        return [shape.as_contour() for shape in self._shapes]

    def as_collapsed_vector(self):
        """
        Returns the shapes as a numpy matrix.
        :return: a Nxd*d numpy matrix containing the shapes
        """
        return np.array([shape.as_collapsed_vector() for shape in self._shapes])

    def get_mean_shape(self):
        """
        Returns the mean shape of the list
        :return: Shape object
        """
        return Shape(np.mean(self.as_numpy_matrix(), axis=0))

    def get_bounding_box(self):
        raw_data = self.as_numpy_matrix()
        return np.array(np.min(np.min(raw_data, axis=0), axis=0), np.max(np.max(raw_data, axis=0), axis=0))

    def concatenate(self, other):
        """
        Returns a new ShapeList concatenating the object and other
        :param other: The other ShapeList
        :return: A ShapeList
        """
        return ShapeList(self._shapes + other.as_list_of_shapes())

    def align(self, tol=1e-7, max_iters=10000, use_transformation_matrix=True, project_to_tangent_space=True):
        """
        Performs Generalized Procrustes Analysis to align a list of shapes
        to a common coordinate system. Implementation based on Appendix A of

        Cootes, Tim, E. R. Baldock, and J. Graham.
        "An introduction to active shape models."
        Image processing and analysis (2000): 223-248.
        :param project_to_tangent_space: Project to tangent space
        :param tol: The convergence threshold for gpa
        (Default: 1e-7)
        :param max_iters: The maximum number of iterations
        permitted (Default: 10000)
        :return: A new shape list with aligned shapes
        """
        if not use_transformation_matrix:
            project_to_tangent_space = False
        if not project_to_tangent_space:
            normalize_over_all_elements = False
        else:
            normalize_over_all_elements = True
        aligned_shapes = [shape.center() for shape in self._shapes]
        mean_shape = aligned_shapes[0].normalize(normalize_over_all_elements)
        for num_iters in range(max_iters):
            for i in range(len(aligned_shapes)):
                aligned_shapes[i] = aligned_shapes[i].align(mean_shape, use_transformation_matrix)
                if project_to_tangent_space:
                    aligned_shapes[i].project_to_tangent_space(mean_shape)
            previous_mean_shape = mean_shape
            mean_shape = Shape(
                np.mean(np.array([shape.as_numpy_matrix() for shape in aligned_shapes]), axis=0)).center().normalize(
                normalize_over_all_elements)
            if np.linalg.norm(mean_shape.as_numpy_matrix() - previous_mean_shape.as_numpy_matrix()) < tol:
                break
        return ShapeList(aligned_shapes)
        


class LineGenerator:
    def __init__(self, point, slope_vector, use_bresenhams_algorithm=True):
        self._point = np.array(point)
        self._slope_vector = slope_vector
        self._use_bresenham = use_bresenhams_algorithm

    def _generate_bresenham(self, num_points, direction):
        delta = float("inf")
        if self._slope_vector[0] != 0:
            delta = float(self._slope_vector[1]) / self._slope_vector[0]
        addendum = np.sign(direction)
        result = []
        error = -1.0
        y = self._point[1]
        x = self._point[0]
        adelta = abs(delta)
        sdelta = np.sign(delta)
        for i in range(num_points+1):
            if i > 0:
                result.append([int(round(x)), int(round(y))])
            if (sdelta > 0 and error + adelta < 0.5) or (sdelta < 0 and error + adelta > -0.5):
                error += adelta
            else:
                if adelta < 1:
                    y += addendum
                else:
                    x += addendum
                error += adelta + sdelta * 1
            if adelta < 1:
                x += addendum
            else:
                y += addendum
        if addendum < 0:
            return list(reversed(result))
        return result

    def _generate_simple(self, num_points, direction):
        unit_slope = np.squeeze(self._slope_vector / np.sqrt(np.sum(self._slope_vector ** 2)))
        result = []
        for increment in range(num_points):
            new_point = (self._point + (direction * unit_slope * (increment + 1)))
            result.append([int(round(new_point[0])), int(round(new_point[1]))])
        return result

    def generate_one_sided(self, num_points, direction):
        if self._use_bresenham:
            points = self._generate_bresenham(num_points, direction)
        else:
            points = self._generate_simple(num_points, direction)
        return points + [[int(round(self._point[0])),int(round(self._point[1]))]]

    def generate_two_sided(self, num_points):
        if self._use_bresenham:
            points_left = self._generate_bresenham(num_points, -1)
            points_right = self._generate_bresenham(num_points, 1)
        else:
            points_left = self._generate_simple(num_points, -1)
            points_right = self._generate_simple(num_points, 1)
        return points_left + [[int(round(self._point[0])),int(round(self._point[1]))]] + points_right