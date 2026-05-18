import zmq
import json

context = zmq.Context()

socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


FUNCTIONS = {
    "add": add,
    "multiply": multiply,
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