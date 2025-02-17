import asyncio
from EyesAndMouthTracker import process_facial_tracking
from Backend import connect_to_vtube_studio

async def main():
    queue = asyncio.Queue()

    tracking_task = asyncio.create_task(process_facial_tracking(queue))
    websocket_task = asyncio.create_task(connect_to_vtube_studio(queue))

    await asyncio.gather(tracking_task, websocket_task)

if __name__ == "__main__":
    asyncio.run(main())
