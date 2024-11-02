from setuptools import setup, find_packages
from os.path import exists, join

entry_points = {
    "console_scripts": [
        "pyimpspec = pyimpspec.cli:main",
    ],
}

dependencies = [
    "Jinja2~=3.1",  # Needed when turning pandas.DataFrames instances into, e.g., LaTeX strings.
    "lmfit~=1.3",  # Needed for performing non-linear fitting.
    "matplotlib~=3.9",  # Needed for the plotting module.
    "numdifftools~=0.9",  # Needed for estimating uncertainties during circuit fitting
    "numpy~=2.1",
    "odfpy~=1.4",  # Needed by pandas for parsing OpenDocument spreadsheet formats.
    "openpyxl~=3.1",  # Needed by pandas for parsing newer Excel files (.xlsx).
    "pandas~=2.2",  # Needed for dealing with various file formats.
    "schemdraw~=0.19",  # Needed to draw circuit diagrams
    "scipy~=1.14",  # Used in the DRT calculations
    "statsmodels~=0.14",  # Used for smoothing (LOWESS) in Z-HIT
    "sympy~=1.13",  # Used to generate expressions for circuits
    "tabulate~=0.9",  # Required by pandas to generate Markdown tables.
    # TODO: The 'xdg' package has been renamed to 'xdg-base-dirs' and changed
    # to only support Python >=3.10. Update at some point in the future.
    "xdg~=6.0",
]

dev_dependencies = [
    "build~=1.2",
    "flake8~=7.1",
    "setuptools~=75.3",
    "sphinx~=8.1",
    "sphinx-rtd-theme~=3.0",
]

optional_dependencies = {
    "cvxopt": "cvxopt~=1.3",  # Used in the DRT calculations (TR-RBF method)
    "kvxopt": "kvxopt~=1.3",  # Fork of cvxopt that may provide wheels for additional platforms
    "dev": dev_dependencies,
}

version = "5.1.0"

if __name__ == "__main__":
    with open("requirements.txt", "w") as fp:
        fp.write("\n".join(dependencies))

    with open("dev-requirements.txt", "w") as fp:
        fp.write("\n".join(dev_dependencies))

    with open("version.txt", "w") as fp:
        fp.write(version)

    assert version.strip != ""
    copyright_notice = ""
    if exists("COPYRIGHT"):
        with open("COPYRIGHT") as fp:
            copyright_notice = fp.read().strip()
        assert copyright_notice != ""

    with open(join("src", "pyimpspec", "version.py"), "w") as fp:
        fp.write(f'{copyright_notice}\n\nPACKAGE_VERSION: str = "{version}"')

    setup(
        name="pyimpspec",
        version=version,
        author="pyimpspec developers",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        include_package_data=True,
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
        entry_points=entry_points,
        install_requires=dependencies,
        extras_require=optional_dependencies,
        python_requires=">=3.10",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering",
        ],
    )
