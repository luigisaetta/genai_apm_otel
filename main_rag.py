"""
RAG REST API

to test APM integration
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# APM integration
from opentelemetry import trace

from langchain_core.messages import HumanMessage, AIMessage

from conversation_manager import ConversationManager
from tracer_singleton import TracerSingleton
from factory import build_rag_chain
from config_reader import ConfigReader
from utils import get_console_logger, sanitize_parameter

# constants
MEDIA_TYPE_TEXT = "text/plain"
MEDIA_TYPE_JSON = "application/json"

#
# Main
#
app = FastAPI()

config = ConfigReader("./config.toml")
VERBOSE = config.find_key("verbose")
# max msgs in conversation
CONV_MAX_MSGS = config.find_key("conv_max_msgs")

logger = get_console_logger()

# Global object to handle conversation history
conversation_manager = ConversationManager(CONV_MAX_MSGS)

# to integrate with OCI APM
TRACER = TracerSingleton.get_instance()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvokeInput(BaseModel):
    """
    class for the body of the request

    query: the request from the user
    """

    query: str


def handle_request(request: InvokeInput, conv_id: str):
    """
    handle the request from invoke
    """
    # build_rag_chain mark a span (see: factory)
    chain = build_rag_chain()

    # get the chat history
    conversation = conversation_manager.get_conversation(conv_id)
    #
    # call the RAG chain
    #
    ai_msg = chain.invoke({"input": request.query, "chat_history": conversation})

    # update the conversation
    conversation_manager.add_message(conv_id, HumanMessage(content=request.query))
    conversation_manager.add_message(conv_id, AIMessage(content=ai_msg["answer"]))

    return ai_msg


#
# HTTP API methods
#
@app.post("/invoke/", tags=["V1"])
@TRACER.start_as_current_span("api.invoke")
def invoke(request: InvokeInput, conv_id: str):
    """
    This function handle the HTTP request

    conv_id: the id of the conversation, to handle chat_history
    """

    # remove eventually any dangerous char
    conv_id = sanitize_parameter(conv_id)

    #
    # This starts the APM trace
    #
    current_span = trace.get_current_span()

    # here we show how to send to APM a value
    current_span.set_attribute("conv_id", conv_id)
    current_span.set_attribute("genai-chat-input", request.query)

    logger.info("Conversation id: %s", conv_id)

    try:
        response = handle_request(request, conv_id)

        # only the text of the response
        answer = response["answer"]

    except Exception as e:
        # to signal error
        answer = f"Error: {str(e)}"

    return Response(content=answer, media_type=MEDIA_TYPE_TEXT)


# to clean up a conversation
@app.delete("/delete/", tags=["V1"])
def delete(conv_id: str):
    """
    delete a conversation
    """
    conv_id = sanitize_parameter(conv_id)

    logger.info("Called delete, conv_id: %s...", conv_id)

    conversation_manager.clear_conversation(conv_id)

    return {"conv_id": conv_id, "messages": []}


if __name__ == "__main__":

    API_HOST = config.find_key("api_host")
    API_PORT = config.find_key("api_port")

    uvicorn.run(host=API_HOST, port=API_PORT, app=app)
