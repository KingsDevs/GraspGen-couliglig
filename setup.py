# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from setuptools import setup, find_packages

setup(
    name="grasp_gen",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        line
        for line in (l.split("#", 1)[0].strip() for l in open("requirements.txt"))
        # skip blanks, inline comments, pip options (-r, --find-links, --extra-index-url),
        # and direct/VCS references (e.g. "pkg @ file://..."), which are not valid
        # install_requires specifiers
        if line and not line.startswith("-") and "@" not in line
    ],
    description="GraspGen",
    author="",
    author_email="",
    license="",
    url="",
    keywords="robotics manipulation learning computer-vision",
    classifiers=[
        "Programming Language :: Python",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
)
