from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qobench.problems.market_share import MarketShareProblem
from qobench.problems.mis import parse_dimacs_graph
from qobench.problems.mkp import parse_mkp_dat_file
from qobench.problems.qap import parse_qap_dat_file


class ParserTests(unittest.TestCase):
    def test_parse_mis_graph(self) -> None:
        content = "\n".join(
            [
                "c comment",
                "p edge 4 3",
                "e 1 2",
                "e 2 3",
                "e 3 4",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "graph.txt"
            path.write_text(content, encoding="utf-8")
            instance = parse_dimacs_graph(path)

        self.assertEqual(instance.num_nodes, 4)
        self.assertEqual(len(instance.edges), 3)
        self.assertIn((0, 1), instance.edges)

    def test_parse_mkp_instance(self) -> None:
        # n=3, m=2, opt=0
        # profits: 10 20 30
        # weights row1: 2 3 4
        # weights row2: 1 2 3
        # capacities: 5 4
        content = "3 2 0 10 20 30 2 3 4 1 2 3 5 4"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mkp.dat"
            path.write_text(content, encoding="utf-8")
            instance = parse_mkp_dat_file(path)

        self.assertEqual(instance.n, 3)
        self.assertEqual(instance.m, 2)
        self.assertEqual(instance.profits, [10, 20, 30])
        self.assertEqual(instance.capacities, [5, 4])

    def test_parse_qap_instance(self) -> None:
        # n=2; flow=[[0,1],[1,0]]; distance=[[0,2],[2,0]]
        content = "2 0 1 1 0 0 2 2 0"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "qap.dat"
            path.write_text(content, encoding="utf-8")
            instance = parse_qap_dat_file(path)

        self.assertEqual(instance.n, 2)
        self.assertEqual(len(instance.flow), 2)
        self.assertEqual(len(instance.flow[0]), 2)
        self.assertEqual(instance.distance[0][1], 2)

    def test_market_share_json_loader(self) -> None:
        content = '{"demands": [[1, 2, 3], [4, 5, 6]]}'
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "market.json"
            path.write_text(content, encoding="utf-8")
            problem = MarketShareProblem()
            instance = problem.load_instance(path, seed=0, target_ratio=0.5)

        self.assertEqual(instance.num_products, 2)
        self.assertEqual(instance.num_retailers, 3)
        self.assertEqual(instance.target_demands, [3, 7])


if __name__ == "__main__":
    unittest.main()
