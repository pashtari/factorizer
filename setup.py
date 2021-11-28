from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="factorizer",
    version="0.0.1",
    author="Pooya Ashtari",
    author_email="pooya.ash@gmail.com",
    description="Factorizer - PyTorch",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pashtari/factorizer",
    project_urls={
        "Bug Tracker": "https://github.com/pashtari/factorizer/issues",
        "Source Code": "https://github.com/pashtari/factorizer",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "machine learning",
        "deep learning",
        "image segmentation",
        "medical image segmentation",
        "Factorizer",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "sympy",
        "scipy",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "torch",
        "torchvision",
        "pytorch-lightning",
        "lightning-bolts",
        "einops",
        "opt_einsum",
        "networkx",
        "plotly",
        "itk",
        "nibabel",
        "monai",
        "timm",
        "performer-pytorch",
    ],
)
