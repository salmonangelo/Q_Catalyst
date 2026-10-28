import React from "react";

export const Esm1vPage: React.FC = () => {
  return (
    <div className="workspace-container">
      <div className="page-header">
        <div className="page-title-row">
          <h2 className="page-title">Stage 3: ESM-1v Mutation-Effect Scores</h2>
          <span className="badge badge-ready">Phase 3 Scheduled</span>
        </div>
        <p className="page-description">
          Compute zero-shot mutation effect scores using masked-marginal log-likelihood ratios.
        </p>
      </div>

      <div className="stub-card">
        <div className="stub-icon">⚡</div>
        <h3 className="stub-title">ESM-1v Scoring Interface Scheduled for Phase 3</h3>
        <p className="stub-desc">
          Core scoring engine in <span className="path-badge">src/qcatalyst/protein_ai/esm_scoring.py</span> is
          implemented with strict offline cache policies and deterministic test mocks.
        </p>

        <div className="stub-details">
          <h4>Engine Capabilities</h4>
          <ul>
            <li>
              <span className="func">ESMScorerConfig(model_name, scoring_strategy)</span> — Model architecture config
            </li>
            <li>
              <span className="func">score_variants_dataframe(df)</span> — Batch variant effect evaluation
            </li>
            <li>
              <span className="func">results/predictions/esm1v_scores.parquet</span> — Output Parquet schema established
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
