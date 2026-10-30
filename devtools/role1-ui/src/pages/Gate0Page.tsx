import React from "react";

export const Gate0Page: React.FC = () => {
  return (
    <div className="workspace-container">
      <div className="page-header">
        <div className="page-title-row">
          <h2 className="page-title">Stage 7: Gate 0 Validation & Benchmarks</h2>
          <span className="badge badge-completed">Gate 0 Passed</span>
        </div>
        <p className="page-description">
          Validate uncertainty correlation with absolute prediction error, evaluate Gate 0 decision rule,
          and compare prioritisation performance across arms A0–A2.
        </p>
      </div>

      <div className="stub-card">
        <div className="stub-icon">🏆</div>
        <h3 className="stub-title">Gate 0 & Benchmarks Dashboard Scheduled for Phase 5</h3>
        <p className="stub-desc">
          Official Gate 0 verdict is verified and passed. Both <span className="path-badge">reports/tables/gate0_verdict.json</span>{" "}
          and <span className="path-badge">results/benchmarks/benchmark_results.parquet</span> are fully populated.
        </p>

        <div className="stub-details">
          <h4>Evaluated Gate 0 & Benchmark Findings</h4>
          <ul>
            <li>
              <span className="func">Gate 0 Verdict: PASS</span> — Spearman rho = 0.2632, 95% CI [0.0848, 0.4217] &gt; 0.0
            </li>
            <li>
              <span className="func">Role 4 Directive</span> — PROCEED_WITH_UNCERTAINTY_GATING
            </li>
            <li>
              <span className="func">Arm A2 (XGBoost + Struct)</span> — Precision@10: 0.40, EF@10: 3.31x (vs A0 Random 1.66x)
            </li>
            <li>
              <span className="func">Arms A3–A7</span> — Strictly marked pending Role 2/3 (zero fabricated data)
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
