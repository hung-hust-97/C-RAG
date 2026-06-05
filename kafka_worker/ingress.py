import asyncio
import logging
import signal
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from kafka_worker.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KMS_EXTRACTING_TOPIC,
    KMS_CHUNKING_TOPIC,
    KMS_EXTRACTING_LEGAL_TOPIC,
    KMS_CHUNKING_LEGAL_TOPIC,
    PIPELINE_GENERAL_EXTRACTION_TOPIC,
    PIPELINE_LEGAL_EXTRACTION_TOPIC,
    PIPELINE_GENERAL_CHUNKING_TOPIC,
    PIPELINE_LEGAL_CHUNKING_TOPIC,
)
import os
from kafka_worker.models import KmsEvent
from lightrag.pipeline.models import PipelineStageEvent, StageName
from lightrag.pipeline.partitioning import create_partition_allocator_from_env
from lightrag.pipeline.coordinator import create_coordinator_from_env
from lightrag.kg.postgres_impl import ClientManager
from lightrag.kg.postgres_impl import to_uuid
from kafka_worker.partitioned_publish import publish_pipeline_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"PROCESSED", "COMPLETED", "FAILED", "DUPLICATED"}


def _normalize_status(status) -> str:
    if hasattr(status, "value"):
        status = status.value
    return str(status or "").strip().upper()


def _is_terminal_status_value(status) -> bool:
    return _normalize_status(status) in TERMINAL_STATUSES


async def is_terminal_status(doc_id: str, workspace_id: str, is_legal: bool) -> bool:
    try:
        ws_uuid = to_uuid(workspace_id)
        db = await ClientManager.get_client()
        async with db.pool.acquire() as conn:
            if is_legal:
                row = await conn.fetchrow(
                    'SELECT process_status as status FROM tbl_legal_document WHERE id = $1 AND workspace_id = $2',
                    int(doc_id), ws_uuid
                )
            else:
                doc_uuid = to_uuid(doc_id)
                row = await conn.fetchrow(
                    'SELECT status FROM tbl_document WHERE document_id = $1 AND workspace_id = $2',
                    doc_uuid, ws_uuid
                )
            if row and _is_terminal_status_value(row["status"]):
                logger.info(
                    "Skipping terminal document doc_id=%s workspace_id=%s status=%s",
                    doc_id,
                    workspace_id,
                    row["status"],
                )
                return True
    except Exception as e:
        logger.error(f"Failed to check terminal status for doc {doc_id}: {e}")
    return False


class IngressAdapter:
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.partition_allocator = create_partition_allocator_from_env()
        self.coordinator = create_coordinator_from_env()
        self.running = False
        self.mapping = {
            KMS_EXTRACTING_TOPIC: (StageName.EXTRACTION, "general", PIPELINE_GENERAL_EXTRACTION_TOPIC),
            KMS_EXTRACTING_LEGAL_TOPIC: (StageName.EXTRACTION, "legal", PIPELINE_LEGAL_EXTRACTION_TOPIC),
            KMS_CHUNKING_TOPIC: (StageName.CHUNKING, "general", PIPELINE_GENERAL_CHUNKING_TOPIC),
            KMS_CHUNKING_LEGAL_TOPIC: (StageName.CHUNKING, "legal", PIPELINE_LEGAL_CHUNKING_TOPIC),
        }
        self.stage_topics = {
            StageName.EXTRACTION: {
                "general": PIPELINE_GENERAL_EXTRACTION_TOPIC,
                "legal": PIPELINE_LEGAL_EXTRACTION_TOPIC,
            },
            StageName.CHUNKING: {
                "general": PIPELINE_GENERAL_CHUNKING_TOPIC,
                "legal": PIPELINE_LEGAL_CHUNKING_TOPIC,
            },
        }

    @staticmethod
    def _canonical_doc_type(mapped_doc_type: str, kms_event: KmsEvent) -> str:
        payload_doc_type = str(kms_event.doc_type or "").lower()
        if (
            mapped_doc_type == "legal"
            or payload_doc_type == "legal"
            or kms_event.legal_document_id is not None
        ):
            return "legal"
        return "general"
        
    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        )
        await self.producer.start()
        
        topics = list(self.mapping.keys())
        self.consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=f"{KAFKA_GROUP_ID}_ingress",
            auto_offset_reset="earliest",
            value_deserializer=lambda x: KmsEvent.model_validate_json(x.decode("utf-8"))
        )
        await self.consumer.start()
        
        self.running = True
        logger.info(f"Ingress adapter started. Listening on: {topics}")
        
        try:
            async for msg in self.consumer:
                if not self.running:
                    break
                await self.process_message(msg.topic, msg.value)
        except Exception as e:
            logger.error(f"Error in ingress consumer loop: {e}")
            raise
        finally:
            await self.stop()
            
    async def process_message(self, topic: str, kms_event: KmsEvent):
        if topic not in self.mapping:
            logger.warning(f"Unknown topic: {topic}")
            return
            
        stage_name, mapped_doc_type, _out_topic = self.mapping[topic]
        doc_type = self._canonical_doc_type(mapped_doc_type, kms_event)
        out_topic = self.stage_topics[stage_name][doc_type]
        doc_id = kms_event.document_id
        if not doc_id and kms_event.legal_document_id is not None:
            doc_id = str(kms_event.legal_document_id)
            
        if not doc_id:
            logger.error(f"Missing doc_id for event: {kms_event}")
            return
            
        # Check if already terminal to prevent rolling back completed/failed docs
        is_legal = (doc_type == "legal")
        if await is_terminal_status(doc_id, kms_event.workspace_id, is_legal):
            logger.info(f"Skipping event for {doc_id} as it is already terminal")
            return
            
        logger.info(f"Ingress mapping: {topic} -> {out_topic} (doc_id={doc_id})")
        
        metadata = kms_event.model_dump()
        metadata["doc_type"] = doc_type
        # Create pipeline event
        pipeline_event = PipelineStageEvent(
            run_id=doc_id,
            workspace_id=kms_event.workspace_id,
            doc_id=doc_id,
            doc_type=doc_type,
            stage=stage_name,
            metadata=metadata
        )
        
        paused_stages_env = os.getenv("PIPELINE_PAUSED_STAGES", "")
        paused_stages = {s.strip().lower() for s in paused_stages_env.split(",") if s.strip()}
        
        if stage_name.value in paused_stages:
            logger.info(
                f"[Ingress] Stage {stage_name.value} is in PIPELINE_PAUSED_STAGES. Pausing document {doc_id}"
            )
            await self.coordinator.register_paused(pipeline_event)
            return

        await publish_pipeline_event(
            self.producer,
            self.partition_allocator,
            out_topic,
            pipeline_event,
        )
        
    async def stop(self):
        self.running = False
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        await self.partition_allocator.finalize()
        await self.coordinator.finalize()
        logger.info("Ingress adapter stopped.")


async def main():
    if not KAFKA_BOOTSTRAP_SERVERS:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS must be set")
        
    adapter = IngressAdapter()
    
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(adapter.stop()))
        
    await adapter.start()

if __name__ == "__main__":
    asyncio.run(main())
