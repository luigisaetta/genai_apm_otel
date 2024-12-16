"""
OracleVS + extension for APM integration
"""

from typing import Dict, List, Any

from langchain_core.documents.base import Document
from langchain_community.vectorstores.oraclevs import OracleVS


SERVICE_NAME = "OracleVS"


class OracleVS4APM(OracleVS):
    """
    Subclass with extension to add tracing for APM
    """

    def similarity_search(
        self, query: str, k: int = 4, filter: Dict[str, Any] | None = None, **kwargs
    ) -> List[Document]:
        """
        Perform a similarity search with APM tracing.

        Args:
            query (str): The query string for the search.
            k (int, optional): The number of top results to return. Defaults to 4.
            filter (Dict[str, Any], optional): A filter to apply to the search. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: A list of documents that match the search criteria.
        """
        return super().similarity_search(query, k=k, filter=filter, **kwargs)
