# SAM3 text-prompt segmentation + grasp-forwarding server environment.
#
# SAM3 needs transformers>=5 (-> a newer huggingface_hub) which is INCOMPATIBLE
# with the ancient diffusers GraspGen depends on. So SAM3 runs in this separate
# image and forwards cleaned point clouds to the GraspGen container over ZMQ.
#
# Code is NOT copied in — the whole repo is bind-mounted to /code at runtime
# (see docker/compose.sam3.yml). PYTHONPATH=/code makes `import grasp_gen`
# resolve (only the lightweight GraspGenClient is imported from it).
#
# Build:
#   docker compose -f docker/compose.sam3.yml build sam3

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# libGL/libglib are needed by opencv even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "transformers>=5" \
        pillow \
        opencv-python-headless \
        numpy \
        scikit-learn \
        pyzmq \
        msgpack \
        msgpack-numpy

WORKDIR /code
ENV PYTHONPATH=/code

EXPOSE 5557

# The launch command (host/port/graspgen target) is supplied by compose.
