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

dependencies = [
    "cvxopt>=1.3.0",  # Used in the DRT calculations
    "lmfit>=1.0.3",  # Needed for performing non-linear fitting.
    "matplotlib>=3.5.2",  # Needed for the plotting module.
    "numpy>=1.23.1",
    "odfpy>=1.4.1",  # Needed by pandas for parsing OpenDocument spreadsheet formats.
    "openpyxl>=3.0.10",  # Needed by pandas for parsing newer Excel files (.xlsx).
    "pandas>=1.4.3",  # Needed for dealing with various file formats.
    "schemdraw>=0.13",  # Needed to draw circuit diagrams
    "scipy>=1.9.0",  # Used in the DRT calculations
    "sympy>=1.10.1",  # Used to generate expressions for circuits
    "tabulate>=0.8.10",  # Required by pandas to generate Markdown tables.
]

optional_dependencies = {
    "cvxpy": "cvxpy>=1.2.1",  # Used in the DRT calculations (TR-RBF method)
}

with open("requirements.txt", "w") as fp:
    fp.write("\n".join(dependencies))

data_files = [
    "COPYRIGHT",
    "CONTRIBUTORS",
    "LICENSES/README.md",
] + licenses

setup(
    name="pyimpspec",
    version="3.1.0",
    author="pyimpspec developers",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    data_files=data_files,
    url="https://vyrjana.github.io/pyimpspec",
    project_urls={
        "Documentation": "https://vyrjana.github.io/pyimpspec/api/",
        "Source Code": "https://github.com/vyrjana/pyimpspec",
        "Bug Tracker": "https://github.com/vyrjana/pyimpspec/issues",
    },
    license="GPLv3",
    description="A package for parsing, validating, analyzing, and simulating impedance spectra.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=dependencies,
    extras_require=optional_dependencies,
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering",
    ],
)
