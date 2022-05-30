# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Inspired by https://github.com/pypa/sampleproject/blob/main/setup.py

setup(
    name="iree-torch",
    version="0.0.1",
    description="PyTorch frontend for IREE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google/iree-torch",
    author="Sean Silva",
    author_email="silvasean@google.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="compiler, deep learning, machine learning, pytorch",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.7, <4",
    install_requires=[
        "iree-compiler",
        "iree-runtime",
        "torch-mlir"
    ],
)
