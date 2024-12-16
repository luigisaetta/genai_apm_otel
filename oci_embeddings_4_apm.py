"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-06-27

This version has been instrumented with zipkin for APM integration
Python Version: 3.11

License: MIT
"""

from langchain_community.embeddings import OCIGenAIEmbeddings

# convention: the name of the superclass
SERVICE_NAME = "OCIGenAIEmbeddings"


#
# extend OCIGenAIEmbeddings adding batching
#
class OCIGenAIEmbeddingsWithBatch(OCIGenAIEmbeddings):
    """
    in addition to  integration with APM
    add batching to OCIEmebeddings
    with Cohere max # of texts is: 96
    """

    # instrumented for integration with APM
    def embed_documents(self, texts):
        """
        in addition to  integration with APM it also add batching
        """
        embeddings = super().embed_documents(texts)

        return embeddings
