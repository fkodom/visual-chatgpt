import os
from distutils.core import setup
from subprocess import getoutput

import setuptools


def get_version_tag() -> str:
    try:
        env_key = "VISUAL_CHATGPT_VERSION".upper()
        version = os.environ[env_key]
    except KeyError:
        version = getoutput("git describe --tags --abbrev=0")

    if version.lower().startswith("fatal"):
        version = "0.0.0"

    return version


extras_require = {
    "test": [
        "black",
        "ruff",
        "mypy>=1.0",
        "pytest",
        "pytest-cov",
        "types-PyYAML",
    ]
}
extras_require["dev"] = ["pre-commit", *extras_require["test"]]
all_require = [r for reqs in extras_require.values() for r in reqs]
extras_require["all"] = all_require


setup(
    name="visual-chatgpt",
    version=get_version_tag(),
    author="Frank Odom",
    author_email="frank.odom.iii@gmail.com",
    url="https://github.com/fkodom/visual-chatgpt",
    packages=setuptools.find_packages(exclude=["tests"]),
    description="For running inference on SmartML models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        # TODO: Separate into extra packages:
        #   - rest
        #   - app
        "einops==0.3.0",
        "fastapi==0.87.0",
        "google-auth==1.35.0",
        "langchain==0.0.101",
        "numpy==1.23.1",
        "omegaconf==2.1.1",
        "opencv-python==4.7.0.72",
        "prometheus-client==0.16.0",
        "prometheus-fastapi-instrumentator==5.9.1",
        "pydantic==1.10.6",
        "streamlit==1.12.1",
        "streamlit-chat",
        "torch==1.12.1",
        "torchvision==0.13.1",
        "transformers==4.26.1",
        "uvicorn[standard]==0.19.0",
        "diffusers",
        "gradio",
        "openai",
    ],
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
