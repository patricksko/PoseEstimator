from setuptools import setup, find_packages

setup(
    name="EstimHelpers",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Detection
        "blenderproc",
        "ultralytics",
        "opencv-python",
        "numpy",

        # Registration
        "open3d",

        # Utils
        "matplotlib",
    ],
    python_requires=">=3.8",
)