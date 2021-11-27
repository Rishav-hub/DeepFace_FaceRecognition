import argparse
import logging
import os
## Imports 
from src.face_embeddings.generate_embeddings import FaceEmbeddings
from src.face_features.extract_attributes import FaceAttributes
from src.face_predictor.generate_prediction import FacePredictor


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'input_log.log'), level=logging.INFO, format=logging_str,
                    filemode="a")
def getFaceEmbeddings():
    """
    Function to get the face embeddings from the images in the given directory
    """
    parser = argparse.ArgumentParser(description='Generate face embeddings')
    parser.add_argument('--faces', default= 50, help='Number of faces that the camera will get')
    parser.add_argument('--detector_backend', type=str, default='mtcnn', help='Face detector to be used')
    parser.add_argument('--config', default='config/config.yaml', help='Path to the config file', type=str)
    parser.add_argument('--params', default='params.yaml', help='Path to the config file', type=str)

    args = vars(parser.parse_args())

    embeddings = FaceEmbeddings(args)
    embeddings.GenerateFaceEmbedding()


def getFaceFeatures():
    parser = argparse.ArgumentParser(description='Generate face features')
    parser.add_argument('--faces', default= 10, help='Number of faces that the camera will get')
    parser.add_argument('--detector_backend', type=str, default='mtcnn', help='Face detector to be used')
    parser.add_argument('--config', default='config/config.yaml', help='Path to the config file', type=str)
    parser.add_argument('--params', default='params.yaml', help='Path to the config file', type=str)

    args = vars(parser.parse_args())

    embeddings = FaceAttributes(args)
    embeddings.ExtractAttributes()

def getFacePrediction():
    """
    Function to get the face embeddings from the images in the given directory
    """
    parser = argparse.ArgumentParser(description='Generate face embeddings')
    parser.add_argument('--faces', default= 50, help='Number of faces that the camera will get')
    parser.add_argument('--detector_backend', type=str, default='mtcnn', help='Face detector to be used')
    parser.add_argument('--config', default='config/config.yaml', help='Path to the config file', type=str)
    parser.add_argument('--params', default='params.yaml', help='Path to the config file', type=str)

    args = vars(parser.parse_args())

    embeddings = FacePredictor(args)
    embeddings.GenerateFacePrediction()


if __name__ == '__main__':

    # try:
    #     logging.info(">>> Stage one Started")
    #     getFaceEmbeddings()
    #     # print(embeddings_list)
    #     logging.info("Stage one Completed >>>>")
    # except Exception as e:
    #     logging.error("Error in Stage one")
    #     logging.error(e)
    #     raise e
    
    # try:
    #     logging.info(">>> Stage two Started")
    #     getFaceFeatures()
    #     logging.info("Stage two Completed >>>>")
    # except Exception as e:
    #     logging.error("Error in Stage two")
    #     logging.error(e)
    #     raise e

    try:
        logging.info(">>> Stage Three Started")
        getFacePrediction()
        # print(embeddings_list)
        logging.info("Stage Three Completed >>>>")
    except Exception as e:
        logging.error("Error in Stage Three")
        logging.error(e)
        raise e

