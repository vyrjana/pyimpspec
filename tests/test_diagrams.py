# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2024 pyimpspec developers
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

from os import makedirs
from os.path import (
    exists,
    join,
)
from tempfile import gettempdir
from typing import List
from pyimpspec import (
    Circuit,
    parse_cdc,
)


if __name__ == "__main__":
    CDCs: List[str] = [
        "R",
        "RL",
        "(RL)",
        "([RL]C)",
        "(R[LC])",
        "(R[LC]W)",
        "(W[(RL)C])Q",
        "RLC",
        "RLCQ",
        "RLCQW",
        "(RLC)",
        "(RLCQ)",
        "(RLCQW)",
        "R(LCQW)",
        "RL(CQW)",
        "RLC(QW)",
        "(RLCQ)W",
        "(RLC)QW",
        "(RL)CQW",
        "R(LCQ)W",
        "R(LC)QW",
        "RL(CQ)W",
        "(R[WQ])",
        "(R[WQ]C)",
        "(R[W(LC)Q])",
        "([LC][RRQ])",
        "(R[WQ])([LC][RRQ])",
        "([RL][CW])",
        "R(RW)",
        "R(RW)C",
        "R(RWL)C",
        "R(RWL)(LQ)C",
        "R(RWL)C(LQ)",
        "R(LQ)C(RWL)",
        "R([RW]Q)C",
        "R(RW)(CQ)",
        "R([RW]Q[LRC])(CQ)",
        "R([RW][L(RQ)C]Q[LRC])(CQ)",
        "R([RW][L(WC)(RQ)C]Q[LRC])(CQ)",
        "(R[LCQW])",
        "(RL[CQW])",
        "(RLC[QW])",
        "(R[LCQ]W)",
        "(R[LC]QW)",
        "(RL[CQ]W)",
        "R(LC)(QW)",
        "(RL)C(QW)",
        "(RL)(CQ)W",
        "(RL)(CQW)",
        "(RLC)(QW)",
        "(R[LC])QW",
        "([RL]C)QW",
        "([RL]CQ)W",
        "([RL]CQW)",
        "([RLC]QW)",
        "([RLCQ]W)",
        "(R[(LC)QW])",
        "(R[L(CQ)W])",
        "(R[LC(QW)])",
        "(R[L(CQW)])",
        "(R[(LCQ)W])",
        "(R[(LC)Q]W)",
        "(R[L(CQ)]W)",
        "(RQ)RWL",
        "RWL(RQ)",
        "(R[QR])(LC)RW",
        "RW(LC)(RQ)",
        "RL(QW)L(RR)(RR)L(RR)C",
        "RL(QW)(L[(RR)(RR)L(RR)C])",
        "RL(QW)(L[(RR)(RR)L(RR)])",
    ]
    output_dir: str = join(gettempdir(), "pyimpspec")
    if not exists(output_dir):
        makedirs(output_dir)
    i: int
    cdc: str
    for i, cdc in enumerate(CDCs):
        circuit: Circuit = parse_cdc(cdc)
        path: str = join(output_dir, f"{i}-{cdc}.svg")
        print(path)
        circuit.to_drawing().save(path)
    # The generated SVG images must be inspected manually.
