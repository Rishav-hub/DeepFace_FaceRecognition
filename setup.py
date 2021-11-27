from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.2",
    author="Rishav-hub",
    description="A small package for dvc ml pipeline Loan Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/Rishav-hub/Loan_Prediction.git",
    author_email="9930046@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        'tensorflow>=1.9.0',
        'keras>=2.2.0',
        'Flask>=1.1.2',
        'mtcnn>=0.1.0',
        'retina-face>=0.0.1',
        'elasticsearch==7.10.0',
        'cmake',
        'pyyaml'
    ]
)