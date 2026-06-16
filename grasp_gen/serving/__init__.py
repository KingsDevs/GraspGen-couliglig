def __getattr__(name):
    if name == "GraspGenZMQServer":
        from grasp_gen.serving.zmq_server import GraspGenZMQServer
        return GraspGenZMQServer
    if name == "GraspGenClient":
        from grasp_gen.serving.zmq_client import GraspGenClient
        return GraspGenClient
    if name == "SAM3Server":
        from grasp_gen.serving.sam3_server import SAM3Server
        return SAM3Server
    if name == "SAM3Client":
        from grasp_gen.serving.sam3_client import SAM3Client
        return SAM3Client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["GraspGenZMQServer", "GraspGenClient", "SAM3Server", "SAM3Client"]
