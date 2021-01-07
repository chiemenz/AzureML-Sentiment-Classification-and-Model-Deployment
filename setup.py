from os import path

import setuptools

ROOT_PATH = path.abspath(path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="automl_vs_hyperdrive",
    version="0.0.1",
    author="Christoph Hiemenz",
    author_email="ch314@gmx.de",
    description="This is a Repo to demonstrate the use of azureml automl and azureml hyperdrive",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chiemenz/automl_vs_hyperdrive",
    packages=['rating_ml_modules'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
