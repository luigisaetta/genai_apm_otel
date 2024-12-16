"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-06-27

This version has been instrumented with Open Telemery for APM integration
Python Version: 3.11

License: MIT
"""

from langchain_community.embeddings import OCIGenAIEmbeddings
from tracer_singleton import TracerSingleton

TRACER = TracerSingleton.get_instance()


#
# extend OCIGenAIEmbeddings to integrate with APM
#
class OCIGenAIEmbeddings4APM(OCIGenAIEmbeddings):
    """
    Subclass to enable addition of annotation
    """

    # instrumented for integration with APM
    @TRACER.start_as_current_span("OCIGenAIEmbeddings.embed_documents")
    def embed_documents(self, texts):
        """
        call the superclass method
        """
        embeddings = super().embed_documents(texts)

        return embeddings
