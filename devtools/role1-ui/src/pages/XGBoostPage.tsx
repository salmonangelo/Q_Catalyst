import React from "react";

export const XGBoostPage: React.FC = () => {
  return (
    <div className="workspace-container">
      <div className="page-header">
        <div className="page-title-row">
          <h2 className="page-title">Stage 5: Supervised PETase Prediction</h2>
          <span className="badge badge-ready">Phase 4 Scheduled</span>
        </div>
        <p className="page-description">
          Train and evaluate primary XGBoost regressor and baseline Ridge comparator on frozen generalization splits.
        </p>
      </div>

      <div className="stub-card">
        <div className="stub-icon">📈</div>
        <h3 className="stub-title">XGBoost Predictor UI Scheduled for Phase 4</h3>
        <p className="stub-desc">
          Supervised predictor in <span className="path-badge">src/qcatalyst/protein_ai/predictor.py</span> has
          been trained on single-site variants and evaluated on 149 held-out multi-mutants.
        </p>

        <div className="stub-details">
          <h4>Evaluated Held-Out Benchmark Metrics (Activity)</h4>
          <ul>
            <li>
              <span className="func">XGBoost MAE: 0.2257</span> vs Ridge: 0.5203 (-56.6% error reduction)
            </li>
            <li>
              <span className="func">XGBoost Spearman rho: 0.5535</span> (p = 2.45e-13) vs Ridge: 0.2493
            </li>
            <li>
              <span className="func">results/predictions/predictions.parquet</span> — 466 predictions saved
            </li>
            <li>
              <span className="func">data/splits/ispetase_activity_single_to_multi.json</span> — Frozen split record
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
