import asyncio
from aiokafka.admin import AIOKafkaAdminClient
from kafka_worker.config import (
    KAFKA_BOOTSTRAP_SERVERS,
)


async def check_consumer_lag():
    print(f"Connecting to Kafka {KAFKA_BOOTSTRAP_SERVERS}...")
    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    await admin.start()

    try:
        groups = await admin.list_consumer_groups()
        print("\nConsumer Groups found:")
        for group in groups:
            print(f" - {group[0]}")

    finally:
        await admin.close()


if __name__ == "__main__":
    if not KAFKA_BOOTSTRAP_SERVERS:
        print("Set KAFKA_BOOTSTRAP_SERVERS first!")
    else:
        asyncio.run(check_consumer_lag())
