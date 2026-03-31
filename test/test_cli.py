from __future__ import annotations

from spatiold.cli import build_cluster_parser, build_parser, build_slim_parser


def test_full_cli_accepts_multi_gene_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--metadata",
            "/tmp/meta.csv",
            "--expression",
            "/tmp/expr.csv",
            "--output-dir",
            "/tmp/out",
            "--multi-gene",
            "--multi-gene-max-genes",
            "5",
            "--multi-gene-pool-size",
            "30",
            "--multi-gene-criterion",
            "bic",
            "--multi-gene-min-improvement",
            "0.1",
        ]
    )
    assert args.multi_gene is True
    assert args.multi_gene_max_genes == 5
    assert args.multi_gene_pool_size == 30
    assert args.multi_gene_criterion == "bic"
    assert args.multi_gene_min_improvement == 0.1


def test_slim_cli_accepts_multi_gene_flag() -> None:
    parser = build_slim_parser()
    args = parser.parse_args(
        [
            "--metadata",
            "/tmp/meta.csv",
            "--expression",
            "/tmp/expr.csv",
            "--output-dir",
            "/tmp/out",
            "--multi-gene",
        ]
    )
    assert args.multi_gene is True


def test_cluster_cli_accepts_multi_gene_flag() -> None:
    parser = build_cluster_parser()
    args = parser.parse_args(
        [
            "--metadata",
            "/tmp/meta.csv",
            "--expression",
            "/tmp/expr.csv",
            "--output-dir",
            "/tmp/out",
            "--multi-gene",
        ]
    )
    assert args.multi_gene is True
