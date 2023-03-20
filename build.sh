#!/bin/bash
# Stop when a non-zero exit code is encountered
set -e

docs_html(){
	sphinx-build "./docs/source" "./docs/build/html"
}

docs_test(){
	sphinx-build -b doctest "./docs/source" "./docs/build/html"
}

docs_latex(){
	sphinx-build -M latexpdf "./docs/source" "./docs/build/latex"
}

validate_tar(){
	echo
	echo "Listing licenses that were bundled in *.tar.gz:"
	# Check if the package license was included
	tar --list -f ./dist/*.tar.gz | grep "LICENSE$"
	# Check if the other licenses were included
	dist="$(tar --list -f ./dist/*.tar.gz | grep "LICENSE-.*\.txt" | sort)"
	repo="$(ls LICENSES | grep "LICENSE-.*.txt" | sort)"
	python -c "from sys import argv; from os.path import basename; dist = set(list(map(basename, argv[1].split('\n')))); repo = set(list(map(basename, argv[2].split('\n')))); assert dist == repo, 'Incorrect set of bundled licenses!'; list(map(print, sorted(dist)))" "$dist" "$repo"
}

validate_wheel(){
	echo
	echo "Listing licenses that were bundled in *.whl:"
	# Check if the package license was included
	unzip -Z1 ./dist/*.whl | grep "LICENSE$"
	# Check if the other licenses were included
	dist="$(unzip -Z1 ./dist/*.whl | grep "LICENSE-.*\.txt" | sort)"
	repo="$(ls LICENSES | grep "LICENSE-.*.txt" | sort)"
	python -c "from sys import argv; from os.path import basename; dist = set(list(map(basename, argv[1].split('\n')))); repo = set(list(map(basename, argv[2].split('\n')))); assert dist == repo, 'Incorrect set of bundled licenses!'; list(map(print, sorted(dist)))" "$dist" "$repo"
}

if [ "$1" == "docs" ]; then
	docs_html
	docs_test
	docs_latex
	exit
fi
if [ "$1" == "docs-html" ]; then
	docs_html
	exit
fi
if [ "$1" == "docs-test" ]; then
	docs_test
	exit
fi
if [ "$1" == "docs-latex" ]; then
	docs_latex
	exit
fi

# Check for uncommitted changes and untracked files
if [ "$(git status --porcelain=v1 | wc -l)" -ne 0 ]; then
	echo "Detected uncommitted changes and/or untracked files!"
	if ! [ "$1" == "override" ]; then
		echo "Continue with the build process anyway by providing 'override' as an argument to this script."
		exit
	fi
fi

# Check for major issues
flake8 . --select=E9,F63,F7,F82 --show-source --statistics
echo "flake8 didn't find any issues..."
echo

# Update non-source code files that should be included
python3 ./pre-build.py

# Build wheel
python3 -m build

# Validate the source and wheel distributions
validate_tar
validate_wheel

# Update documentation
# - The contents of './docs/build/html' should be committed to the gh-pages branch
# - ./docs/build/latex/latex/pyimpspec.pdf should be uploaded as an attachment to a release
echo
echo "Generating documentation..."
# Generate HTML, run tests, and finally generate PDF
docs_html
docs_test
docs_latex

# Copy documentation assets
python3 ./post-build.py

# Everything should be okay
echo
echo "Finished!!!"
