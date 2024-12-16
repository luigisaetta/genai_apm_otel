"""
factory.py

integrated with APM tracing
"""

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# these are the extension to add APM tracing
from oci_embeddings_4_apm import OCIGenAIEmbeddings4APM
from chatocigenai_4_apm import ChatOCIGenAI4APM
from factory_vector_store import get_vector_store
from prompts_library import CONTEXT_Q_PROMPT, QA_PROMPT
from tracer_singleton import TracerSingleton
from config_reader import ConfigReader
from utils import get_console_logger

from config_private import COMPARTMENT_ID

SERVICE_NAME = "Factory"

config = ConfigReader("./config.toml")
VERBOSE = config.find_key("verbose")
AUTH_TYPE = config.find_key("auth_type")

logger = get_console_logger()

# for APM integration
TRACER = TracerSingleton.get_instance()


def get_embed_model():
    """
    get the Embeddings Model
    """

    embed_model = OCIGenAIEmbeddings4APM(
        auth_type=AUTH_TYPE,
        model_id=config.find_key("embed_model"),
        service_endpoint=config.find_key("embed_endpoint"),
        compartment_id=COMPARTMENT_ID,
    )

    return embed_model


def get_llm():
    """
    Build and return the LLM client
    """

    model_id = config.find_key("llm_model")
    max_tokens = config.find_key("max_tokens")
    temperature = config.find_key("temperature")

    if VERBOSE:
        logger.info("%s as ChatModel...", model_id)

    llm = ChatOCIGenAI4APM(
        # this example uses api_key
        auth_type=AUTH_TYPE,
        model_id=model_id,
        service_endpoint=config.find_key("endpoint"),
        compartment_id=COMPARTMENT_ID,
        is_stream=True,
        model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
    )

    return llm


@TRACER.start_as_current_span("build_rag_chain")
def build_rag_chain():
    """
    build the entire rag chain with Langchain LCEL
    """
    embed_model = get_embed_model()

    v_store = get_vector_store(embed_model=embed_model)

    # num of docs returned from semantic search
    top_k = config.find_key("top_k")

    retriever = v_store.as_retriever(search_kwargs={"k": top_k})

    chat_model = get_llm()

    # using prompt defined in prompt_library
    # 3/07 modified for chat interface
    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, CONTEXT_Q_PROMPT
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, QA_PROMPT)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
