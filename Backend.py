import asyncio
import websockets
import json

VTS_WS_URL = "ws://localhost:8001"
AUTH_TOKEN = "your_auth_token"


async def send_to_vtube_studio(websocket, queue):
    """Send EAR and MAR data to VTube Studio."""
    while True:
        try:
            data = await queue.get()  # Get tracking data from the queue

            expression_data = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "ExpressionRequest",
                "messageType": "ExpressionActivationRequest",
                "data": {
                    "expressions": [
                        {"id": "EyeLeftX", "value": data["ear_left"]},
                        {"id": "EyeRightX", "value": data["ear_right"]},
                        {"id": "MouthOpen", "value": data["mar"]},
                    ]
                },
            }
            await websocket.send(json.dumps(expression_data))

        except Exception as e:
            print(f"Error sending data to VTube Studio: {e}")


async def connect_to_vtube_studio(queue):
    """Connect to VTube Studio via WebSocket and authenticate."""
    try:
        async with websockets.connect(VTS_WS_URL) as websocket:
            print("Connected to VTube Studio!")

            # Authentication
            if AUTH_TOKEN:
                auth_message = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "AuthRequest",
                    "messageType": "AuthenticationRequest",
                    "data": {
                        "pluginName": "FacialTracker",
                        "pluginDeveloper": "YourName",
                        "authenticationToken": AUTH_TOKEN,
                    },
                }
                await websocket.send(json.dumps(auth_message))
                response = await websocket.recv()
                print("Authentication Response:", response)

            # Start sending tracking data
            await send_to_vtube_studio(websocket, queue)

    except Exception as e:
        print(f"WebSocket connection error: {e}")
