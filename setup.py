import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "earthorbit",
    version = "0.1.0",
    author = "Léo Giroud",
    author_email = "grd.leo.99@gmail.com",
    description = "Lightweight package for orbit calculations for satellites orbiting Earth; Tools for generating commands and orientation.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/leogarithm",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Development Status :: 4 - Beta'
    ],
    install_requires= [
        "numpy",
        "arrow",
        "matplotlib",
        "requests",
        "pyquaternion"
    ],
    python_requires='>=3.6'
)