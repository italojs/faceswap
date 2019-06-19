import sys
import cv2
import dlib
import numpy
from imutils import face_utils

facil_points = {
    "face": list(range(17, 68)),
    "mouth": list(range(48, 61)),
    "brow": {
        "right": list(range(17, 22)),
        "left": list(range(22, 27))
    },
    "eye": {
        "right": list(range(36, 42)),
        "left": list(range(42, 48))
    },
    "nose": list(range(27, 35)),
    "jaw": list(range(0, 17))
}

MASK_POINTS = [
    facil_points['eye']['left'] +
    facil_points['eye']['right'] + 
    facil_points['brow']['left'] + 
    facil_points['brow']['right'] +  
    facil_points['nose'] + 
    facil_points['mouth']
]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, (255, 255, 255), 16, 0)

def get_mask(im, landmarks, feather_amount=11):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in MASK_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im])
    im = im.transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (feather_amount, feather_amount), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (feather_amount, feather_amount), 0)
    return im
    
def linear_transformation(points1, points2):
    # https://en.wikipedia.org/wiki/Singular_value_decomposition
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)

    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    
    points1 /= s1
    points2 /= s2

    U, _, Vt = numpy.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def warpAffine(im, M, dshape):
    # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
    output = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output

def merge_colours(im1, im2, landmarks1, color_correct_blur_frac=0.6):
    eye = facil_points['eye']
    blur_amount = int(color_correct_blur_frac * numpy.linalg.norm(
                              numpy.mean(landmarks1[eye['left']], axis=0) -
                              numpy.mean(landmarks1[eye['right']], axis=0)))

    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    divided = im1_blur.astype(numpy.float64) / im2_blur.astype(numpy.float64)
    im2_blur = (im2.astype(numpy.float64) * divided)
    
    return im2_blur

def set_scale(im, scale=1):
    return cv2.resize(im, (im.shape[1] * scale,
                            im.shape[0] * scale))

def get_landmarks_points(im):
    rects = detector(im, 1)

    print('Found {} face'.format(len(rects)))
    if len(rects) < 2:
        print(" is missing two faces. skipping.")    # copy and skip a frame if it's missing two faces
        return None

    landmarks1 = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    landmarks2 = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[1]).parts()])
    return  (landmarks1, landmarks2)

def swap_face(im, landmarks, output):
    M = linear_transformation(landmarks[0][tuple(MASK_POINTS)], 
                            landmarks[1][tuple(MASK_POINTS)])
                            
    mask = get_mask(im, landmarks[1]) 

    warped_mask = warpAffine(mask, M, im.shape)

    mask2 = get_mask(im, landmarks[0])
    combined_mask = numpy.max([mask2, warped_mask],
                        axis=0)
                        
    warped_im2 = warpAffine(im, M, im.shape)
    warped_corrected_im2 = merge_colours(im, warped_im2, landmarks[0])

    return output * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask # apply first mask

def main():  
    im = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    im = set_scale(im)
    landmarks = get_landmarks_points(im)

    if landmarks is None:
        sys.exit(1)
        
    output = im.copy()
    output = swap_face(im, (landmarks[0], landmarks[1]), output)
    output = swap_face(im, (landmarks[1], landmarks[0]), output)

    cv2.imwrite('output.jpg', output)

if __name__ == '__main__':
    main()
