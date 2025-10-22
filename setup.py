from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-optimization-framework",
    version="2.0.0",
    author="N Sri Krishna",
    author_email="your.email@example.com",
    description="Intelligent ML model optimization with automatic constraint solving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/N-SriKrishna/ml-optimization-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ml-optimize=examples.real_world.02_optimize_yolov8:main",
        ],
    },
)
