# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2023 pyimpspec developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import (
    List,
    IO,
)
from os import (
    makedirs,
    walk,
)
from os.path import (
    exists,
    isfile,
    isdir,
    join,
)
from shutil import rmtree


def update_file(src: str, dst: str):
    if not isfile(src):
        return
    src_contents: str = ""
    fp: IO
    with open(src, "r") as fp:
        src_contents = fp.read()
    if isfile(dst):
        with open(dst, "r") as fp:
            if fp.read() == src_contents:
                return
    with open(dst, "w") as fp:
        fp.write(src_contents)


def copy_additional_files(files):
    src_dir: str = "."
    dst_dir: str = join(".", "src", "pyimpspec")
    licenses_dir: str = join(dst_dir, "LICENSES")
    if not isdir(licenses_dir):
        makedirs(licenses_dir)
    path: str
    for path in files:
        update_file(join(src_dir, path), join(dst_dir, path))


if __name__ == "__main__":
    data_files: List[str] = [
        "CHANGELOG.md",
        "CONTRIBUTORS",
        "COPYRIGHT",
        "LICENSE",
        "README.md",
    ]
    files: List[str]
    for _, _, files in walk("LICENSES"):
        data_files.extend(map(lambda _: join("LICENSES", _), files))
        break
    assert all(map(lambda _: isfile(_), data_files))
    copy_additional_files(data_files)
    # Remove old dist files
    dist_output: str = "./dist"
    if exists(dist_output):
        rmtree(dist_output)
    # Remove old documentation files to force a rebuild
    docs_output: str = "./docs/build"
    if exists(docs_output):
        rmtree(docs_output)
