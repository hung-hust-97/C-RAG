"""
Entrypoint for the Kafka worker package.

Usage:
    python -m kafka_worker.pipeline_main
    python -m kafka_worker.ingress
"""

from kafka_worker.pipeline_main import run

if __name__ == "__main__":
    run()
