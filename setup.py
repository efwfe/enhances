from setuptools import setup, find_packages

setup(
    name="data-enhance",
    version="0.1.0",
    description="Custom Albumentations transforms for background manipulation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "albumentations>=1.3.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
    ],
)
