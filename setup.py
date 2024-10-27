from setuptools import setup, find_packages

setup(
    name="qiskit_state_evolution_recorder",
    version="0.1.0",
    description="Simple module allowing to record animations to trace changes in qubit states for arbitrary quantum circuits.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Dawid Ciepiela",
    author_email="71898979+sarumaj@users.noreply.github.com",
    url="https://github.com/sarumaj/qiskit-state-evolution-recorder",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.9.2",
        "matplotlib-inline==0.1.7",
        "numpy==2.1.1",
        "qiskit==1.2.4",
        "qiskit-aer==0.15.1",
        "pillow==11.0.0"
    ],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    license="BSD-3-Clause",
    python_requires=">=3.6, <3.12",
)