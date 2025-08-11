from setuptools import setup, find_packages

setup(
    name="ObjectDetection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "blenderproc",
        "ultralytics",
        "opencv-python",
        "numpy",
        "open3d",
        "matplotlib"
    ],
)