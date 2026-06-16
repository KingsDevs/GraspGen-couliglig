# GraspGen ZMQ inference serving environment.
#
# Builds on the graspgen:latest base (built via `bash docker/build.sh`) and adds
# the ZMQ serving deps. Code is NOT copied in — the whole repo is bind-mounted to
# /code at runtime (see docker/compose.sam3.yml), so the image only carries the
# environment. PYTHONPATH=/code makes `import grasp_gen` resolve from the mount.
#
# Build:
#   docker compose -f docker/compose.sam3.yml build graspgen
#   # or: docker build -f docker/serve.dockerfile -t graspgen-server:latest .

FROM graspgen:latest

RUN pip install --no-cache-dir pyzmq msgpack msgpack-numpy

WORKDIR /code
ENV PYTHONPATH=/code

# The launch command (gripper config, port, scale) is supplied by compose.
EXPOSE 5556
