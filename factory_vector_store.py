"""
Author: Luigi Saetta
Date created: 2024-05-20
Date last modified: 2024-05-23

Usage:
    This module handles the creation of the Vector Store 
    used in the RAG chain, based on config

Python Version: 3.11
"""

import logging
import oracledb

from langchain_community.vectorstores.utils import DistanceStrategy

from config_reader import ConfigReader
from oraclevs_4_apm import OracleVS4APM

from config_private import DB_USER, DB_PWD, DSN, TNS_ADMIN, WALLET_PWD

#
# Configs
#
config = ConfigReader("./config.toml")
VERBOSE = config.find_key("verbose")
SERVICE_NAME = "Factory Vector Store"


def get_db_connection():
    """
    get a connection to db
    """
    logger = logging.getLogger("ConsoleLogger")

    # common params
    conn_parms = {"user": DB_USER, "password": DB_PWD, "dsn": DSN, "retry_count": 3}

    # connection to ADB, needs wallet
    conn_parms.update(
        {
            "config_dir": TNS_ADMIN,
            "wallet_location": TNS_ADMIN,
            "wallet_password": WALLET_PWD,
        }
    )

    if VERBOSE:
        logger.info("")
        logger.info("Connecting as USER: %s to DSN: %s", DB_USER, DSN)

    try:
        return oracledb.connect(**conn_parms)
    except oracledb.Error as e:
        logger.error("Database connection failed: %s", str(e))
        raise


def get_vector_store(vector_store_type, embed_model):
    """
    vector_store_type: can be 23AI
    embed_model an object wrapping the model used for embeddings
    return a Vector Store Object
    """

    logger = logging.getLogger("ConsoleLogger")

    v_store = None

    if vector_store_type == "23AI":
        try:
            connection = get_db_connection()

            v_store = OracleVS4APM(
                client=connection,
                table_name=config["vector_store"]["collection_name"],
                distance_strategy=DistanceStrategy.COSINE,
                embedding_function=embed_model,
            )
        except oracledb.Error as e:
            err_msg = "A DB error occurred in get_vector_store: " + str(e)
            logger.error(err_msg)
        except Exception as e:
            # Catch all other exceptions
            err_msg = "An unexpected error occurred in get_vector_store: " + str(e)
            logger.error(err_msg)
            logger.error(e)

    return v_store
