import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2


def imshow2(img, width=12, height=12):
    plt.figure(figsize=(width, height))
    try:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    except:
        plt.imshow(img)


def plot_line(data,title=None,xlabel=None,ylabel=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(data)), data)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_many_lines(datalines):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for data in datalines:
        ax.plot(range(len(data)), data)
    plt.show()


def plot_shapes(shape_list,labels= None,as_lines=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, len(shape_list)))
    ci = 0
    for index,shape in enumerate(shape_list):
        x_list, y_list = zip(*(shape.as_list_of_points()))
        if labels is None:
            lbl = str(index)
        else:
            lbl = labels[index]
        if as_lines:
            ax.plot((-1 * np.array(x_list)).tolist(), (-1 * np.array(y_list)).tolist(),label=lbl)
        else:
            ax.scatter((-1 * np.array(x_list)).tolist(), (-1 * np.array(y_list)).tolist(), color=colors[ci],label=lbl)
        ci += 1
    plt.legend()
    plt.show()


def overlay_shapes_on_image(img, shape_list):
    im = img.copy()
    cv2.polylines(im, np.int32([np.round(shape.as_numpy_matrix()) for shape in shape_list]), True, (0, 255, 255))
    return im


def overlay_points_on_image(img, points,width=10,color = (0, 255, 255),something_that_was_one=1):
    im = img.copy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    points = np.uint32(np.round(points))
    for point in points:
        cv2.circle(im, (point[0], point[1]), width, color, something_that_was_one)
    return im
