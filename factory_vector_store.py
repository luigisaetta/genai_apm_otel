"""
Author: Luigi Saetta
Date created: 2024-05-20
Date last modified: 2024-05-23

Usage:
    This module handles the creation of the Vector Store 
    used in the RAG chain, based on config

Python Version: 3.11
"""

import oracledb

from langchain_community.vectorstores.utils import DistanceStrategy

from config_reader import ConfigReader
from tracer_singleton import TracerSingleton
from oraclevs_4_apm import OracleVS4APM
from utils import get_console_logger

from config_private import DB_USER, DB_PWD, DSN, TNS_ADMIN, WALLET_PWD

#
# Configs
#
config = ConfigReader("./config.toml")
VERBOSE = config.find_key("verbose")
SERVICE_NAME = "Factory Vector Store"

# for APM integration
TRACER = TracerSingleton.get_instance()

logger = get_console_logger()


@TRACER.start_as_current_span("get_db_connection")
def get_db_connection():
    """
    get a connection to db

    this function works if the DB is ADB
    """

    conn_parms = {
        "user": DB_USER,
        "password": DB_PWD,
        "dsn": DSN,
        "retry_count": 3,
        "config_dir": TNS_ADMIN,
        "wallet_location": TNS_ADMIN,
        "wallet_password": WALLET_PWD,
    }

    if VERBOSE:
        logger.info("")
        logger.info("Connecting as USER: %s to DSN: %s", DB_USER, DSN)

    try:
        return oracledb.connect(**conn_parms)
    except oracledb.Error as e:
        logger.error("Database connection failed: %s", str(e))
        raise


@TRACER.start_as_current_span("get_vector_store")
def get_vector_store(embed_model):
    """
    embed_model an object wrapping the model used for embeddings
    return a Vector Store Object
    """

    v_store = None

    try:
        db_conn = get_db_connection()

        v_store = OracleVS4APM(
            client=db_conn,
            table_name=config.find_key("collection_name"),
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
