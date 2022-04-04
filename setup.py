from setuptools import setup, find_packages

setup(
    name="pyimpspec",
    version="0.1.3",
    author="pyimpspec developers",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url="https://github.com/vyrjana/pyimpspec",
    project_urls={
        "Bug Tracker": "https://github.com/vyrjana/pyimpspec/issues",
    },
    license="GPLv3",
    description="A package for parsing, validating, analyzing, and simulating impedance spectra.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "lmfit>=1.0.3",  # Needed for performing non-linear fitting.
        "matplotlib>=3.5.1",  # Needed for the plotting module.
        "numpy>=1.21.5",
        "odfpy>=1.4.1",  # Needed by pandas for parsing OpenDocument spreadsheet formats.
        "openpyxl>=3.0.9",  # Needed by pandas for parsing newer Excel files (.xlsx).
        "pandas>=1.3.5",  # Needed for dealing with various file formats.
        "sympy>=1.9",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
)
