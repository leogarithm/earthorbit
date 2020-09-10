import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "earthorbit",
    version = "0.0.1",
    author = "Léo Giroud",
    author_email = "grd.leo.99@gmail.com",
    description = "Lightweight package for orbit calculations for objets orbiting Earth.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/leogarithm",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires= [
        "numpy",
        "arrow"
    ],
    python_requires='>=3.6',
)