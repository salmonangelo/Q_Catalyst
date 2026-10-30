"""Unit and contract tests for Q-Catalyst benchmarking and metrics (Feature 8)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcatalyst.evaluation.benchmarks import (
    load_quantum_results,
    run_benchmark_suite,
    save_benchmark_artifacts,
    score_arm_a0_random,
    score_arm_a1_esm1v,
    score_arm_a2_xgboost_structural,
)
from qcatalyst.evaluation.metrics import (
    compute_bootstrap_ci,
    dcg_at_k,
    enrichment_factor_at_k,
    evaluate_all_ranking_metrics,
    kendall_tau_correlation,
    ndcg_at_k,
    precision_at_k,
    spearman_rank_correlation,
)


def test_precision_at_k():
    """Verify Precision@K calculation on deterministic arrays."""
    # 5 items: true labels [0.5, 1.2, 0.8, 1.5, 0.2]
    # Scores: [10, 9, 8, 7, 6] -> ranked order is indices 0, 1, 2, 3, 4
    # Top 2 are indices 0 (0.5 <= 1.0) and 1 (1.2 > 1.0) -> hits = 1/2 = 0.5
    y_true = np.array([0.5, 1.2, 0.8, 1.5, 0.2])
    y_score = np.array([10.0, 9.0, 8.0, 7.0, 6.0])

    p2 = precision_at_k(y_true, y_score, k=2, threshold=1.0)
    assert p2 == 0.5

    p4 = precision_at_k(y_true, y_score, k=4, threshold=1.0)
    # Top 4 are indices 0, 1, 2, 3 -> hits are indices 1 (1.2) and 3 (1.5) -> 2/4 = 0.5
    assert p4 == 0.5

    # If scores perfectly prioritize hits:
    y_score_perf = np.array([1.0, 10.0, 2.0, 9.0, 0.5])
    # Top 2 are indices 1 and 3 -> hits = 2/2 = 1.0
    assert precision_at_k(y_true, y_score_perf, k=2, threshold=1.0) == 1.0

    # Edge cases
    assert precision_at_k(y_true, y_score, k=0) == 0.0
    assert precision_at_k(np.array([]), np.array([]), k=5) == 0.0


def test_enrichment_factor_at_k():
    """Verify Enrichment Factor: EF@K = Precision@K / Base_Rate."""
    y_true = np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # 1 hit out of 10 -> base_rate = 0.1
    # Model puts the 1 hit at top rank
    y_score = np.array([1, 10, 2, 3, 4, 5, 6, 7, 8, 9])  # index 1 has score 10

    # Top 1 has precision 1.0
    ef1 = enrichment_factor_at_k(y_true, y_score, k=1, threshold=1.0)
    assert np.isclose(ef1, 10.0)  # 1.0 / 0.1 = 10.0x enrichment

    # When no hits exist in dataset:
    y_none = np.zeros(10)
    assert enrichment_factor_at_k(y_none, y_score, k=2) == 0.0


def test_ndcg_at_k():
    """Verify NDCG@K is 1.0 for perfect ranking and strictly lower for reversed ranking."""
    y_true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    # Perfect ranking
    y_score_ideal = np.array([50, 40, 30, 20, 10])
    ndcg_ideal = ndcg_at_k(y_true, y_score_ideal, k=5)
    assert np.isclose(ndcg_ideal, 1.0)

    # Completely reversed ranking
    y_score_rev = np.array([10, 20, 30, 40, 50])
    ndcg_rev = ndcg_at_k(y_true, y_score_rev, k=5)
    assert 0.0 < ndcg_rev < ndcg_ideal


def test_correlations_spearman_and_kendall():
    """Verify Spearman and Kendall correlations on monotonic and constant arrays."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    sp_r, sp_p = spearman_rank_correlation(x, y)
    kt_r, kt_p = kendall_tau_correlation(x, y)
    assert np.isclose(sp_r, 1.0)
    assert np.isclose(kt_r, 1.0)

    # Inverted
    sp_r_inv, _ = spearman_rank_correlation(x, -y)
    assert np.isclose(sp_r_inv, -1.0)

    # Constant array handles safely
    sp_c, sp_cp = spearman_rank_correlation(np.ones(5), y)
    assert sp_c == 0.0
    assert sp_cp == 1.0


def test_compute_bootstrap_ci():
    """Verify bootstrap confidence interval brackets the point estimate."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0.5, 2.0, size=50)
    y_score = y_true + rng.normal(0, 0.2, size=50)

    pt, low, high = compute_bootstrap_ci(
        y_true,
        y_score,
        metric_func=lambda yt, ys: precision_at_k(yt, ys, k=10, threshold=1.0),
        n_bootstrap=500,
        ci_level=0.95,
        seed=42,
    )

    assert 0.0 <= low <= pt <= high <= 1.0


def test_score_arms_functions():
    """Verify scoring logic for A0, A1, and A2."""
    n = 20
    s_a0 = score_arm_a0_random(n, random_seed=42)
    assert len(s_a0) == n
    assert np.all(np.isfinite(s_a0))

    y_pred = np.linspace(0.5, 1.5, n)
    struct_risk = np.linspace(0.1, 0.9, n)
    s_a2 = score_arm_a2_xgboost_structural(y_pred, struct_risk, alpha_struct=0.10)
    assert len(s_a2) == n
    assert np.all(s_a2 >= y_pred)  # boosted by non-negative structural risk


def test_benchmark_suite_real_dataset_integrity():
    """Verify execution of run_benchmark_suite on actual project data."""
    variants_path = Path("data/curated/variants.parquet")
    preds_path = Path("results/predictions/predictions.parquet")
    uncert_path = Path("results/uncertainty/uncertainty.parquet")

    if not uncert_path.exists():
        pytest.skip("results/uncertainty/uncertainty.parquet not found.")

    df_results = run_benchmark_suite(
        variants_path=variants_path,
        predictions_path=preds_path,
        uncertainty_path=uncert_path,
        target_name="activity",
        enzyme="IsPETase",
        n_bootstrap=200,
        random_seed=42,
    )

    assert not df_results.empty
    arms = df_results["arm_id"].unique()
    assert "A0" in arms
    assert "A1" in arms
    assert "A2" in arms
    assert "A3" in arms  # Pending arm

    # Check evaluated arms
    eval_df = df_results[df_results["status"] == "evaluated"]
    assert len(eval_df) > 0
    assert np.all(np.isfinite(eval_df["metric_value"]))

    # Check pending arms (A3–A7 must NOT have invented numbers)
    pending_df = df_results[df_results["status"] == "pending_role2_role3_results"]
    assert len(pending_df) == 5
    assert np.all(np.isnan(pending_df["metric_value"]))


def test_save_benchmark_artifacts(tmp_path: Path):
    """Verify export of Parquet, CSV, and Markdown benchmark reports."""
    variants_path = Path("data/curated/variants.parquet")
    preds_path = Path("results/predictions/predictions.parquet")
    uncert_path = Path("results/uncertainty/uncertainty.parquet")

    if not uncert_path.exists():
        pytest.skip("results/uncertainty/uncertainty.parquet not found.")

    df = run_benchmark_suite(
        variants_path=variants_path,
        predictions_path=preds_path,
        uncertainty_path=uncert_path,
        n_bootstrap=100,
        random_seed=42,
    )

    parquet_file = tmp_path / "benchmark_results.parquet"
    csv_file = tmp_path / "benchmark_summary.csv"
    report_file = tmp_path / "benchmark_report.md"

    save_benchmark_artifacts(df, parquet_file, csv_file, report_file)

    assert parquet_file.exists() and parquet_file.stat().st_size > 0
    assert csv_file.exists() and csv_file.stat().st_size > 0
    assert report_file.exists() and report_file.stat().st_size > 0

    # Verify Parquet reload
    reloaded = pd.read_parquet(parquet_file)
    assert len(reloaded) == len(df)
    assert "precision@10" in reloaded["metric_name"].values


# -------------------------------------------------------------
# Role 3 Quantum Results Loader Unit and Contract Tests
# -------------------------------------------------------------
def test_load_quantum_results_success():
    """Verify successful loading and validation of the exported WT quantum results Parquet."""
    quantum_file = Path("results/quantum/quantum_results.parquet")
    if not quantum_file.exists():
        pytest.skip("results/quantum/quantum_results.parquet does not exist.")

    df = load_quantum_results(quantum_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    row = df.iloc[0]

    # Verify WT identity and role
    assert row["variant_id"] == "VAR_WT"
    assert row["mutation"] == "WT"
    assert row["active_space"] == "CAS(4e,4o)"

    # Verify finite energy values
    casci_energy = float(row["quantum_casci_reference_energy_hartree"])
    vqe_energy = float(row["vqe_energy_hartree"])
    reported_error = float(row["absolute_vqe_error_hartree"])
    assert np.isfinite(casci_energy)
    assert np.isfinite(vqe_energy)
    assert np.isfinite(reported_error)

    # Verify numerical error consistency within 1e-10 Ha tolerance
    expected_error = abs(vqe_energy - casci_energy)
    assert abs(reported_error - expected_error) <= 1e-10

    # Test default path parameter
    df_default = load_quantum_results()
    assert len(df_default) == 1
    assert df_default.iloc[0]["variant_id"] == "VAR_WT"


def test_load_quantum_results_missing_file(tmp_path: Path):
    """Verify load_quantum_results raises FileNotFoundError when file does not exist."""
    missing_path = tmp_path / "nonexistent_results.parquet"
    with pytest.raises(FileNotFoundError, match="Quantum results Parquet file not found"):
        load_quantum_results(missing_path)


def test_load_quantum_results_missing_columns(tmp_path: Path):
    """Verify load_quantum_results raises ValueError when required contract columns are missing."""
    base_file = Path("results/quantum/quantum_results.parquet")
    if not base_file.exists():
        pytest.skip("results/quantum/quantum_results.parquet does not exist.")

    base_df = pd.read_parquet(base_file)
    # Drop one required contract column
    corrupt_df = base_df.drop(columns=["vqe_energy_hartree"])
    bad_path = tmp_path / "missing_vqe_column.parquet"
    corrupt_df.to_parquet(bad_path)

    with pytest.raises(ValueError, match="missing required contract columns"):
        load_quantum_results(bad_path)


def test_load_quantum_results_duplicate_wt_records(tmp_path: Path):
    """Verify load_quantum_results raises ValueError when multiple WT records are present."""
    base_file = Path("results/quantum/quantum_results.parquet")
    if not base_file.exists():
        pytest.skip("results/quantum/quantum_results.parquet does not exist.")

    base_df = pd.read_parquet(base_file)
    duplicate_df = pd.concat([base_df, base_df], ignore_index=True)
    dup_path = tmp_path / "duplicate_wt.parquet"
    duplicate_df.to_parquet(dup_path)

    with pytest.raises(ValueError, match="Expected exactly one 'VAR_WT' reference record"):
        load_quantum_results(dup_path)


def test_load_quantum_results_missing_wt_record(tmp_path: Path):
    """Verify load_quantum_results raises ValueError when no VAR_WT record is found."""
    base_file = Path("results/quantum/quantum_results.parquet")
    if not base_file.exists():
        pytest.skip("results/quantum/quantum_results.parquet does not exist.")

    base_df = pd.read_parquet(base_file)
    mutant_only_df = base_df.copy()
    mutant_only_df["variant_id"] = "VAR_S92A"
    mutant_path = tmp_path / "no_wt.parquet"
    mutant_only_df.to_parquet(mutant_path)

    with pytest.raises(ValueError, match="must contain a 'VAR_WT' reference record"):
        load_quantum_results(mutant_path)


def test_load_quantum_results_inconsistent_energy_error(tmp_path: Path):
    """Verify load_quantum_results raises ValueError when absolute_vqe_error_hartree is inconsistent."""
    base_file = Path("results/quantum/quantum_results.parquet")
    if not base_file.exists():
        pytest.skip("results/quantum/quantum_results.parquet does not exist.")

    base_df = pd.read_parquet(base_file)
    bad_error_df = base_df.copy()
    # Inconsistent error: true diff is ~2.4e-5 Ha, set to 0.5 Ha
    bad_error_df["absolute_vqe_error_hartree"] = 0.5
    bad_err_path = tmp_path / "bad_error.parquet"
    bad_error_df.to_parquet(bad_err_path)

    with pytest.raises(ValueError, match="Inconsistent absolute_vqe_error_hartree"):
        load_quantum_results(bad_err_path)


def test_load_quantum_results_non_finite_energies(tmp_path: Path):
    """Verify load_quantum_results raises ValueError when energies are NaN or infinite."""
    base_file = Path("results/quantum/quantum_results.parquet")
    if not base_file.exists():
        pytest.skip("results/quantum/quantum_results.parquet does not exist.")

    base_df = pd.read_parquet(base_file)
    nan_df = base_df.copy()
    nan_df["vqe_energy_hartree"] = np.nan
    nan_path = tmp_path / "nan_vqe.parquet"
    nan_df.to_parquet(nan_path)

    with pytest.raises(ValueError, match="must be finite numbers"):
        load_quantum_results(nan_path)


def test_load_quantum_results_invalid_dtype(tmp_path: Path):
    """Verify load_quantum_results raises TypeError when data types are not numeric."""
    base_file = Path("results/quantum/quantum_results.parquet")
    if not base_file.exists():
        pytest.skip("results/quantum/quantum_results.parquet does not exist.")

    base_df = pd.read_parquet(base_file)
    bad_type_df = base_df.copy()
    bad_type_df["ansatz_parameters"] = "twenty_six"
    bad_type_path = tmp_path / "bad_dtype.parquet"
    bad_type_df.to_parquet(bad_type_path)

    with pytest.raises(TypeError, match="must have a numeric integer data type"):
        load_quantum_results(bad_type_path)

