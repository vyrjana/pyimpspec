from setuptools import setup, find_packages
from os import walk
from os.path import join


licenses = []
for _, _, files in walk("LICENSES"):
    licenses.extend(
        list(
            map(
                lambda _: join("LICENSES", _),
                filter(lambda _: _.startswith("LICENSE-"), files),
            )
        )
    )

setup(
    name="pyimpspec",
    version="1.1.0",
    author="pyimpspec developers",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    data_files=[
        "COPYRIGHT",
        "CONTRIBUTORS",
        "LICENSES/README.md",
    ]
    + licenses,
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
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering",
    ],
)
