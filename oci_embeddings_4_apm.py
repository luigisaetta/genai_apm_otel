"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-06-27

This version has been instrumented with Open Telemery for APM integration
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
    Subclass to enable addition of annotation
    """

    # instrumented for integration with APM
    def embed_documents(self, texts):
        """
        in addition to  integration with APM it also add batching
        """
        embeddings = super().embed_documents(texts)

        return embeddings
