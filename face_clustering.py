import sys
import os
import dlib
import glob
import yaml
import pandas as pd
import numpy as np
from math import sqrt

def euclidean_dist(vector_x, face_descriptor):
    if len(vector_x) != len(face_descriptor):
        raise Exception('Vectors must be same dimensions')

    x = np.array(vector_x)
    y = np.array(face_descriptor)
    return sum((x[dim] - y[dim]) ** 2 for dim in range(len(x)))

class face_clustering:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

    def clustering(self):

        face_clustering_result = []
        face_embedding_result = []
        face_id = 0

        descriptors = []
        images = []

        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(self.config['shape_predictor'])
        facerec = dlib.face_recognition_model_v1(self.config['face_recognition_model_v1'])

        for f in glob.glob(self.config['image_input_folder']):
            self.logger.info("Processing file: {}".format(f))
            img = dlib.load_rgb_image(f)

            # Ask the detector to find the bounding boxes of each face. The 1 in the
            # second argument indicates that we should upsample the image 1 time. This
            # will make everything bigger and allow us to detect more faces.
            dets = detector(img, 1)
            self.logger.info("Number of faces detected: {}".format(len(dets)))

            # Now process each face we found.
            fcr = {}
            fcr["img_file"] = f
            fcr["no_of_faces"] = len(dets)
            for k, d in enumerate(dets):
                fcr_ed = {}
                fcr_ed['face'] = ''.join(['face_id_', str(face_id)])
                
                # Get the landmarks/parts for the face in box d.
                shape = sp(img, d)

                # Compute the 128D vector that describes the face in img identified by
                # shape.  
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                fcr_ed['embedding'] = face_descriptor
                fcr_ed['cluster'] = ''

                descriptors.append(face_descriptor)
                images.append((img, shape))

                # d: (d.left(), d.top(), d.right(), d.bottom())
                # fcr["face_rec"] = d
                fcr['face_cluster_in_image'] = []

                face_embedding_result.append(fcr_ed)

                face_id = face_id + 1

            face_clustering_result.append(fcr)

        #Now let's cluster the faces.  
        labels = dlib.chinese_whispers_clustering(descriptors, self.config['cw_clustering_threshold'])
        num_classes = len(set(labels))

        face_sum = 0
        pic_no = 0

        for i in range(len(labels)):
            face_embedding_result[i]['cluster'] = labels[i]

            if i + 1 <= face_sum + face_clustering_result[pic_no]['no_of_faces']:
                face_clustering_result[pic_no]['face_cluster_in_image'].append(labels[i])
            else:
                pic_no = pic_no + 1
                face_sum = face_sum + face_clustering_result[pic_no - 1]['no_of_faces']
                face_clustering_result[pic_no]['face_cluster_in_image'].append(labels[i])
            
        face_clustering_result_df = pd.DataFrame(face_clustering_result)
        face_embedding_result_df = pd.DataFrame(face_embedding_result)

        clusters = [[] for _ in range(num_classes)]
        for i, pair in enumerate(images):
            clusters[labels[i]].append(pair)

        return face_clustering_result_df, face_embedding_result_df, clusters

    def save_image_face_cluster_result(self, face_clustering_result_df, face_embedding_result_df):

        face_embedding_result_df['embedding'] = face_embedding_result_df['embedding'] = face_embedding_result_df['embedding'].map(lambda x: list(np.array(x)))
        
        face_clustering_result_df.to_csv(self.config['face_clustering_output'], index=False)
        face_embedding_result_df.to_csv(self.config['face_embedding_output'], index=False)

    def save_cluster_result(self, clusters):
        try:
            for i, cluster in enumerate(clusters):
                cluster_folder_path = os.path.join(self.config['face_output_folder'], str(i))
                if len(cluster) > self.config['cluster_component_number_threshold']:
                    if not os.path.isdir(cluster_folder_path):
                        os.makedirs(cluster_folder_path)
                    for j, pair in enumerate(cluster):
                        img, shape = pair
                        dlib.save_face_chip(img, shape, os.path.join(cluster_folder_path, 'face_{}'.format(j)), size=150, padding=0.25)
                    self.logger.info("faces in cluser {} saved in {}".format(str(i), cluster_folder_path))
            return True

        except Exception as e:
            self.logger.error("error happened {}".format(e))
            return False

    def output_image_list(self, face_clustering_result_df, face_embedding_result_df, img_input_dir):

        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(self.config['shape_predictor'])
        facerec = dlib.face_recognition_model_v1(self.config['face_recognition_model_v1'])

        face_typical_embedding_df = face_embedding_result_df.groupby('cluster').head(1).reset_index(drop=True)

        img = dlib.load_rgb_image(img_input_dir)

        dets = detector(img, 1)
        for k, d in enumerate(dets):    
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)

            # Compute the 128D vector that describes the face in img identified by
            # shape.  
            face_descriptor = facerec.compute_face_descriptor(img, shape)

        face_typical_embedding_df['distance'] = face_typical_embedding_df['embedding'].map(lambda x: euclidean_dist(x, face_descriptor = face_descriptor))

        output_cluster = face_typical_embedding_df.iloc[face_typical_embedding_df['distance'].idxmin()]['cluster']

        return face_clustering_result_df[[output_cluster in i for i in face_clustering_result_df['face_cluster_in_image']]].img_file.tolist()

        