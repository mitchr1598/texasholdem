import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="texasholdem1598",
    version="1.0.3",
    description="A package for playing texas holdem poker",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mitchr1598/texasholdem",
    author="mitchr1598",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    packages=["texasholdem"],
    python_requires=">=3.9.0",
    install_requires=[
          'playingcards1598',
      ],
)
