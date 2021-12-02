from src.deepface.detectors import FaceDetector
from src.deepface import DeepFace
import sys
import numpy as np
import cv2
import os
import logging
from src.utils.all_utils import print_exams_average, read_yaml
from datetime import datetime
from src.deepface.commons import functions
from src.face_simmilarity.store_embeddings import storeEmbeddings

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'input_log.log'), level=logging.INFO, format=logging_str,
                    filemode="a")


class FaceEmbeddings:

    def __init__(self, args):
        self.args = args
        self.facedetector = FaceDetector
        self.deepface = DeepFace
        self.detector = 'mtcnn'
        # self.config_path = 'config/config.yaml'

    def GenerateFaceEmbedding(self):
        config_path = read_yaml(self.args['config'])
        face_detector = self.facedetector.build_model(self.args['detector_backend'])

        cap = cv2.VideoCapture(0)
        faces1= 0
        frames = 0
        max_faces = int(self.args["faces"])

        model = self.deepface.build_model("Facenet")
        embeddings_list = []
        while faces1 < max_faces:
            _, frame = cap.read()
            frames += 1
            if frames % 3 == 0:
                faces = FaceDetector.detect_faces(face_detector, self.args['detector_backend'], frame, align=False)
                try:
                    img = functions.preprocess_face(frame, target_size= (model.input_shape[1], model.input_shape[2]), detector_backend= 'mtcnn', enforce_detection= False)
                    embedding = model.predict(img)[0].tolist()
                    embeddings_list.append(embedding)
                    
                except Exception as e:
                    logging.exception(e)
                    continue

                for face, (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (67,67,67), 2)

            #         print('Face detected')
                faces1 += 1
            cv2.imshow("Face detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        embeddings_list = print_exams_average(embeddings_list)
        # print(embeddings_list)
        print(len(embeddings_list))
        

        ###########Store Embeddings################
        logging.info('>>> Storing Embeddings')
        storeEmbeddings(self.args['config'], self.args['params'], embeddings_list)
        logging.info('Embeddings Stored >>>')



