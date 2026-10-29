from setuptools import setup, find_packages

setup(
    name="guardrailnet",
    version="1.0.0",
    description="End-to-End Deep Learning System for Road Guardrail Detection, Condition Classification & Safety Risk Telemetry",
    author="Deep Learning Engineering Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python-headless>=4.8.0",
        "streamlit>=1.30.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=9.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)

