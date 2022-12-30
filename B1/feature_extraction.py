import os
import cv2
import dlib
import numpy as np

from tqdm import tqdm
from keras_preprocessing import image

global basedir, image_paths, predictor

basedir_train = "./Datasets/dataset_AMLS_22-23/cartoon_set"
basedir_test = "./Datasets/dataset_AMLS_22-23_test/cartoon_set"
images_dir_train = os.path.join(basedir_train, "img")
images_dir_test = os.path.join(basedir_test, "img")
labels_filename = "labels.csv"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./A1/shape_predictor_68_face_landmarks.dat")
# Label = "Train"


def extract_features_labels(Label):
    # if __name__ == "__main__":

    def shape_to_np(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def rect_to_bb(rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    def run_dlib_shape(image):
        # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
        # load the input image, resize it, and convert it to grayscale
        resized_image = image.astype("uint8")

        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype("uint8")

        # detect faces in the grayscale image
        rects = detector(gray, 1)
        num_faces = len(rects)

        if num_faces == 0:
            return None, resized_image

        face_areas = np.zeros((1, num_faces))
        face_shapes = np.zeros((136, num_faces), dtype=np.int64)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            temp_shape = predictor(gray, rect)
            temp_shape = shape_to_np(temp_shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)],
            #   (x, y, w, h) = face_utils.rect_to_bb(rect)
            (x, y, w, h) = rect_to_bb(rect)
            face_shapes[:, i] = np.reshape(temp_shape, [136])
            face_areas[0, i] = w * h
        # find largest face and keep
        dlibout = np.reshape(
            np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2]
        )

        return dlibout, resized_image

    def extract_features_from_img(eyes_feature, image):

        max_x = np.max(eyes_feature[0])
        min_x = np.min(eyes_feature[0])
        min_y = np.min(eyes_feature[1])
        max_y = np.max(eyes_feature[1])

        eyes = np.ones((len(range(min_y, max_y)), len(range(min_x, max_x)), 3))
        eyes = image[min_y:max_y+1, min_x:max_x+1]

        return eyes

    def extract_eyes_features(img_path, features, eye_start_points, eye_end_point):

        image = cv2.imread(img_path)

        # Extract eyes Features from dlib detector
        features_left_eye = features[eye_start_points:eye_end_point][:6]
        features_right_eye = features[eye_start_points:eye_end_point][6:]

        # Find the region which encompassed by those eyes feawtures points
        features_left_eye = extract_features_from_img(
            np.transpose(features_left_eye), image)
        features_right_eye = extract_features_from_img(
            np.transpose(features_right_eye), image)

        # dim_1 = min(features_left_eye.shape[0], features_right_eye.shape[0])
        # dim_2 = min(features_left_eye.shape[1], features_right_eye.shape[1])

        # features_left_eye = np.resize(features_left_eye, (dim_1, dim_2, 3))
        # features_right_eye = np.resize(features_right_eye, (dim_1, dim_2, 3))
        # np.vstack((features_left_eye, features_right_eye))

        return features_left_eye, features_right_eye

    def resize_feature(features_left_eye, features_right_eye):

        dim_1_list, dim_2_list = [], []

        # get the dimension of eyes region matrix
        for i in range(len(features_left_eye)):
            dim_1_list.append(features_left_eye[i].shape[0])
            dim_1_list.append(features_right_eye[i].shape[0])

        for j in range(len(features_left_eye)):
            dim_2_list.append(features_left_eye[i].shape[1])
            dim_2_list.append(features_right_eye[i].shape[1])

        dim_1 = int(np.mean(dim_1_list))
        dim_2 = int(np.mean(dim_2_list))

        # resize the region with the avg dimension
        features_eyes = []
        for i in range(len(features_left_eye)):
            resized_left_eye_features = np.resize(
                features_left_eye[i], (dim_1, dim_2, 3))
            resized_right_eye_features = np.resize(
                features_right_eye[i], (dim_1, dim_2, 3))
            resized_eyes_features = np.vstack(
                (resized_left_eye_features, resized_right_eye_features))
            features_eyes.append(resized_eyes_features)

        return features_eyes

    def extract_features_labels(Label):
        """
        This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
        It also extracts the gender label for each image.
        :return:
            landmark_features:  an array containing 68 landmark points for each image in which a face was detected
            all_labels: an array containing both gender label (male=0 and female=1) and smiling label (not smiling = 0 and smiling = 1)
                        for each image in which a face was detected
        """
        if Label == "Train":
            basedir = basedir_train
            images_dir = images_dir_train
            total_img = len(os.listdir(images_dir_train))

        elif Label == "Test":
            basedir = basedir_test
            images_dir = images_dir_test
            total_img = len(os.listdir(images_dir_test))

        else:
            raise TypeError("Dataset is unclear labelled")

        image_paths = [os.path.join(images_dir, l)
                       for l in os.listdir(images_dir)]
        target_size = None
        labels_file = open(os.path.join(basedir, labels_filename), "r")
        lines = labels_file.readlines()
        eye_color_labels = [line.split("\t")[1] for line in lines[1:]]
        eye_start_points, eye_end_point = 36, 48
        features_left_eye, features_right_eye = [], []

        face_shape_labels = [line.split("\t")[2] for line in lines[1:]]
        face_start_points, face_end_point = 0, 17
        face_features = []

        if os.path.isdir(images_dir):
            all_features = {'Face': [], 'Eyes': []}
            all_labels = [[] * 1 for _ in range(2)]
            with tqdm(total=total_img, unit="img", desc="Loading " + Label + " Dataset") as pbar:
                for img_path in image_paths:
                    file_name = img_path.split("\\")[-1].split(".")[0]
                    # load image
                    img = image.img_to_array(image.load_img(img_path,
                                                            target_size=target_size,
                                                            interpolation="bicubic"))
                    features, _ = run_dlib_shape(img)
                    if features is not None:
                        face_features.append(
                            features[face_start_points:face_end_point])
                        features_left, features_right = extract_eyes_features(
                            img_path, features, eye_start_points, eye_end_point)
                        features_left_eye.append(features_left)
                        features_right_eye.append(features_right)

                        all_labels[0].append(
                            int(face_shape_labels[int(file_name)]))
                        all_labels[1].append(
                            int(eye_color_labels[int(file_name)]))
                    pbar.update(1)

        all_features['Face'].append(np.array(face_features))
        # dim_1 = min(eye_features[i].shape[0] for i in range(len(eye_features)))
        # dim_2 = min(eye_features[i].shape[1] for i in range(len(eye_features)))
        # for i in range(len(eye_features)):
        #     eye_features[i]= np.resize(eye_features[i], (dim_1, dim_2, 3))
        eye_features = resize_feature(features_left_eye, features_right_eye)
        # a = features_train['Eyes'][0][1] to see diff image features
        all_features['Eyes'].append(np.array(eye_features))

        all_features = {k: np.array(v) for k, v in all_features.items()}
        # simply converts the -1 into 0, so male=0 and female=1; non-smiling=0 and smiling=1
        all_labels = (np.array(all_labels) + 1) / 2

        return all_features, all_labels

    # features_train, train_labels = extract_features_labels("Train")
