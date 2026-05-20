import base64
import zmq
import json
import numpy as np

context = zmq.Context()

socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

def _decode_image_payload(image_payload):
    if isinstance(image_payload, dict):
        if "data" in image_payload:
            image = np.asarray(
                image_payload["data"],
                dtype=image_payload.get("dtype", np.uint8),
            )
            if "shape" in image_payload:
                image = image.reshape(image_payload["shape"])
            return image

        if "base64" in image_payload:
            encoded_image = image_payload["base64"]
        else:
            raise ValueError("image dict must include 'data' or 'base64'")
    elif isinstance(image_payload, str):
        encoded_image = image_payload
    else:
        return np.asarray(image_payload, dtype=np.uint8)

    image_bytes = base64.b64decode(encoded_image)
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)

    import cv2

    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image payload")
    return image


def view_image_point_cloud_point(image, point_cloud, point):
    try:
        import cv2

        image = _decode_image_payload(image)
        cv2.imshow("RPC image", image)
        cv2.waitKey(1)
    except Exception as exc:
        print("Could not display image with OpenCV:", exc)

    point_cloud = np.asarray(point_cloud)
    point = np.asarray(point)
    print("Received point cloud:", point_cloud.shape)
    print("Received 3D point:", point.tolist())

    return "ok"


FUNCTIONS = {
    "view_image_point_cloud_point": view_image_point_cloud_point,
}

print("RPC Server started...")

while True:
    message = socket.recv_json()

    func_name = message["function"]
    args = message.get("args", [])

    print("Received:", message)

    if func_name in FUNCTIONS:
        result = FUNCTIONS[func_name](*args)

        socket.send_json({
            "success": True,
            "result": result
        })
    else:
        socket.send_json({
            "success": False,
            "error": "Function not found"
        })
