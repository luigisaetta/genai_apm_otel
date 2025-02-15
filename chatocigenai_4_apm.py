"""
For integration with APM
"""

from typing import Any, List

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage
from langchain_core.language_models import LanguageModelInput
from langchain_community.chat_models import ChatOCIGenAI
from opentelemetry import trace

from tracer_singleton import TracerSingleton


TRACER = TracerSingleton.get_instance()


class ChatOCIGenAI4APM(ChatOCIGenAI):
    """
    Extension for integration with Application Performance Monitoring (APM).
    """

    @TRACER.start_as_current_span("ChatOCIGenAI.invoke")
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: List[str] | None = None,
        **kwargs: Any
    ) -> BaseMessage:
        """
        Invokes the ChatOCIGenAI model with APM integration.

        Args:
            input (LanguageModelInput): The input for the language model.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            stop (List[str], optional): List of stop words. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseMessage: The output from the language model.
        """

        # here we show how to send to SPM a value
        current_span = trace.get_current_span()
        current_span.set_attribute("llm_model", self.model_id)

        output = super().invoke(input, config=config, stop=stop, **kwargs)

        # len in chars of input, output
        llm_model_input_len = len(str(input))
        llm_model_output_len = len(str(output.content))

        current_span.set_attribute("llm_model_input_len", llm_model_input_len)
        current_span.set_attribute("llm_model_output_len", llm_model_output_len)

        return output

    @TRACER.start_as_current_span("stream")
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: List[str] | None = None,
        **kwargs: Any
    ):
        """
        Invokes the ChatOCIGenAI model with APM integration.

        Args:
            input (LanguageModelInput): The input for the language model.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            stop (List[str], optional): List of stop words. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            stream generator
        """
        current_span = trace.get_current_span()
        current_span.set_attribute("llm_model", self.model_id)

        generator = super().stream(input, config=config, stop=stop, **kwargs)

        return generator
