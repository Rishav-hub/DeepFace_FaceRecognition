from src.deepface.detectors import FaceDetector
from src.deepface import DeepFace
from src.deepface.commons import functions
import sys
import numpy as np
import cv2
import pandas as pd
import os
import logging
from datetime import datetime
from src.deepface.extendedmodels import Age
from src.utils.all_utils import read_yaml


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'input_log.log'), level=logging.INFO, format=logging_str,
                    filemode="a")


class FaceAttributes:
    def __init__(self, args):
        self.args = args
        self.facedetector = FaceDetector
        self.deepface = DeepFace
        self.detector = 'mtcnn'
    
    def ExtractAttributes(self):
        config_path = read_yaml(self.args['config'])
        face_detector = self.facedetector.build_model(self.args['detector_backend'])
        cap = cv2.VideoCapture(0)
        faces_1 = 0
        frames = 0
        max_faces = self.args['faces']
        ##################### Load model ##########################
        model = self.deepface.build_model("Facenet")

        emotion_model = self.deepface.build_model('Emotion')
        logging.info("Emotion model loaded")

        age_model = self.deepface.build_model('Age')
        logging.info("Age model loaded")

        gender_model = self.deepface.build_model('Gender')
        logging.info("Gender model loaded")



        while faces_1 < max_faces:
            ret, frame = cap.read()
            frames += 1
            
            dtString = str(datetime.now().microsecond)
        #     if not (os.path.exists(path)):
        #         os.makedirs(path)
            try:
                faces = FaceDetector.detect_faces(face_detector, self.args['detector_backend'], frame, align=False)
                face_index = 0
                detected_faces = []
                img = frame.copy()
                for face, (x, y, w, h) in faces:
                    if w > 130: #discard small detected faces
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
                        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

                        detected_faces.append((x,y,w,h))
                        detected_faces_final = detected_faces.copy()
        #                     face_index = face_index + 1
                        for detected_face in detected_faces_final:
                            x = detected_face[0]; y = detected_face[1]
                            w = detected_face[2]; h = detected_face[3]

                            cv2.rectangle(frame, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image


                            custom_face = img[y:y+h, x:x+w]

                            gray_img = functions.preprocess_face(custom_face, target_size=(48, 48), detector_backend= self.args['detector_backend'],grayscale = True, enforce_detection= False)
        #                         print(gray_img)
                            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                            emotion_predictions = emotion_model.predict(gray_img)[0,:]
                            sum_of_predictions = emotion_predictions.sum()
                            mood_items = []
                            for i in range(0, len(emotion_labels)):
                                    mood_item = []
                                    emotion_label = emotion_labels[i]
                                    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                                    mood_item.append(emotion_label)
                                    mood_item.append(emotion_prediction)
                                    mood_items.append(mood_item)

                            emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
                            emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
                            overlay = img.copy()
                            opacity = 0.4
                            pivot_img_size = 112
                            resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]
                            freeze_img = img.copy()
                            if x+w+pivot_img_size < resolution_x:
        #                         print('if 1')

                                cv2.rectangle(frame
                                    #, (x+w,y+20)
                                    , (x+w,y)
                                    , (x+w+pivot_img_size, y+h)
                                    , (64,64,64),cv2.FILLED)

                                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

                            elif x-pivot_img_size > 0:
        #                         print('else 1')
                                cv2.rectangle(img
                                        , (x-pivot_img_size,y)
                                        , (x, y+h)
                                        , (64,64,64),cv2.FILLED)   
                                cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                            for index, instance in emotion_df.iterrows():
                                emotion_label = "%s " % (instance['emotion'])
                                emotion_score = instance['score']/100

                                bar_x = 35 #this is the size if an emotion is 100%
                                bar_x = int(bar_x * emotion_score)

                                if x+w+pivot_img_size < resolution_x:
        #                             print('if2')

                                    text_location_y = y + 20 + (index+1) * 20
                                    text_location_x = x+w

                                    if text_location_y < y + h:
                                        cv2.putText(frame, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #                                 print('text')
                                        cv2.rectangle(frame
                                            , (x+w+70, y + 13 + (index+1) * 20)
                                            , (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                            , (255,255,255), cv2.FILLED)

                                elif x-pivot_img_size > 0:
        #                             print('else2')

                                    text_location_y = y + 20 + (index+1) * 20
                                    text_location_x = x-pivot_img_size

                                    if text_location_y <= y+h:
                                        cv2.putText(frame, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                        cv2.rectangle(frame
                                            , (x-pivot_img_size+70, y + 13 + (index+1) * 20)
                                            , (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                            , (255,255,255), cv2.FILLED)
                            
                            #----------------------------------------------------------------------------------------
                            # Age Prediction
                            #----------------------------------------------------------------------------------------
                            face_224 = functions.preprocess_face(custom_face, target_size =(224, 224), grayscale = False, enforce_detection = False, detector_backend = self.args['detector_backend'])
                            age_predictions = age_model.predict(face_224)[0,:]
                            apparent_age = Age.findApparentAge(age_predictions)
                            
                            #----------------------------------------------------------------------------------------
                            # Gender Prediction
                            #----------------------------------------------------------------------------------------
                            gender_prediction = gender_model.predict(face_224)[0,:]

                            if np.argmax(gender_prediction) == 0:
                                gender = "W"
                            elif np.argmax(gender_prediction) == 1:
                                gender = "M"

                            analysis_report = str(int(apparent_age))+" "+gender
                            
                            #-----------------------------------------------------------------------------------------
                            # Draw Boxes Age and Gender
                            #-----------------------------------------------------------------------------------------
                            info_box_color = (46,200,255)

                            #top
                            if y - pivot_img_size + int(pivot_img_size/5) > 0:

                                triangle_coordinates = np.array( [
                                    (x+int(w/2), y)
                                    , (x+int(w/2)-int(w/10), y-int(pivot_img_size/3))
                                    , (x+int(w/2)+int(w/10), y-int(pivot_img_size/3))
                                ] )

                                cv2.drawContours(frame, [triangle_coordinates], 0, info_box_color, -1)

                                cv2.rectangle(frame, (x+int(w/5), y-pivot_img_size+int(pivot_img_size/5)), (x+w-int(w/5), y-int(pivot_img_size/3)), info_box_color, cv2.FILLED)

                                cv2.putText(frame, analysis_report, (x+int(w/3.5), y - int(pivot_img_size/2.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

                            #bottom
                            elif y + h + pivot_img_size - int(pivot_img_size/5) < resolution_y:

                                triangle_coordinates = np.array( [
                                    (x+int(w/2), y+h)
                                    , (x+int(w/2)-int(w/10), y+h+int(pivot_img_size/3))
                                    , (x+int(w/2)+int(w/10), y+h+int(pivot_img_size/3))
                                ] )

                                cv2.drawContours(frame, [triangle_coordinates], 0, info_box_color, -1)

                                cv2.rectangle(frame, (x+int(w/5), y + h + int(pivot_img_size/3)), (x+w-int(w/5), y+h+pivot_img_size-int(pivot_img_size/5)), info_box_color, cv2.FILLED)

                                cv2.putText(frame, analysis_report, (x+int(w/3.5), y + h + int(pivot_img_size/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
                            
            
                faces_1 += 1
            except Exception as e:
                print(e)
                continue
            cv2.imshow("Face detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

