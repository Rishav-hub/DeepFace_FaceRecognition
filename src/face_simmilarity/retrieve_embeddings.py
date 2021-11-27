from elasticsearch import Elasticsearch
import yaml
from src.utils.all_utils import read_yaml
import logging
import os
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'input_log.log'), level=logging.INFO, format=logging_str,
                    filemode="a")


def retrieveEmbeddings(path_to_config, path_to_params, average_embeddings):
    config_path = read_yaml(path_to_config)
    param_path = read_yaml(path_to_params)

    es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    mapping = {
    "mappings": {
        "properties": {
            "title_vector":{
                "type": "dense_vector",
                "dims": 128
            },
            "title_name": {"type": "keyword"}
            }
        }
    }
    # logging.info(">>>> Creating index")
    # es.indices.create(index= config_path['index_name'], body=mapping)
    # logging.info("Index created >>>")
    # name = config_path['USER_NAME']
    index = param_path['index']
    logging.info("Index number: {}".format(index))
    # average_embeddings = print_exams_average(embeddings_list)
    query = {
    "size": 5,
    "query": {
    "script_score": {
        "query": {
            "match_all": {}
        },
        "script": {
            "source": "cosineSimilarity(params.queryVector, 'title_vector') + 1.0",
# #             "source": "1 / (1 + l2norm(params.queryVector, 'title_vector'))", #euclidean distance
            "params": {
                "queryVector": list(average_embeddings)
            }
        }
    }
    }}
    logging.info(">>>> Serching embeddings .......")
    res = es.search(index=config_path['index_name'], body=query)
    logging.info("Serching embeddings ....... >>>>")

    return res
