"""
Tracer Singleton

    to support integration with OCI APM
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.resources import Resource

from config_reader import ConfigReader
from config_private import APM_PUBLIC_KEY
from utils import get_console_logger


# this class is used to handle NO tracing
class NoopSpanExporter(SpanExporter):
    """
    NoOp exporter for OpenTelemetry
    """

    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


config_tracing = ConfigReader("./config_tracing.toml")


class TracerSingleton:
    """
    Singleton to handle tracing with OpenTelemetry to OCI APM
    """

    _instance = None

    @staticmethod
    def get_instance():
        """
        return the single instance
        """
        if TracerSingleton._instance is None:
            # Inizializza il tracer al primo accesso
            TracerSingleton._instance = TracerSingleton._init_tracer()
        return TracerSingleton._instance

    @staticmethod
    def _init_tracer():
        """
        Init tracer for APM integration
        """

        trace_enable = config_tracing.find_key("trace_enable")
        apm_endpoint = config_tracing.find_key("apm_endpoint")
        service_name = config_tracing.find_key("service_name")
        tracer_name = config_tracing.find_key("tracer_name")

        logger = get_console_logger()

        # Configura il tracer
        resource = Resource(attributes={"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if trace_enable:
            # Configure OTLP if tracing is enabled
            logger.info("Enabling APM tracing...")

            exporter = OTLPSpanExporter(
                endpoint=apm_endpoint,
                headers={"authorization": f"dataKey {APM_PUBLIC_KEY}"},
            )
        else:
            # Usa un NoOpSpanExporter per scartare le trace
            exporter = NoopSpanExporter()

        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)

        # Restituisce il tracer configurato
        return trace.get_tracer(tracer_name)

    @staticmethod
    def get_tracer_name():
        """
        get the name of the configured tracer
        """
        return config_tracing.find_key("tracer_name")
