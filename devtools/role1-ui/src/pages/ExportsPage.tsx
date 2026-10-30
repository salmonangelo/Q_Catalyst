import React from "react";
import { ProcessBadge } from "../components/ProcessBadge";

export const ExportsPage: React.FC = () => {
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-title">
          <div className="page-title-row">
            <h1>Reports & Exports Centre</h1>
            <ProcessBadge status="not_implemented" label="Phase 6 Roadmap" />
          </div>
          <p className="page-subtitle">
            Catalog, inspect, and download generated artifacts, benchmark parquet files,
            uncertainty tables, and Gate 0 audit summaries.
          </p>
        </div>
      </div>

      <div className="roadmap-card">
        <h3>Feature Planned for Phase 6</h3>
        <p>
          This developer workbench page is scheduled for Phase 6 implementation.
          The core exports and artifacts are currently persisted on disk in approved directories:
        </p>

        <div className="code-block" style={{ marginTop: "1rem", marginBottom: "1rem" }}>
          <code>reports/tables/petgym_file_manifest.csv</code><br />
          <code>reports/tables/petgym_csv_audit.csv</code><br />
          <code>reports/tables/petgym_audit_summary.md</code><br />
          <code>data/curated/variants.parquet</code><br />
          <code>results/predictions/esm1v_scores.parquet</code><br />
          <code>results/embeddings/esm2_t12_35M_UR50D_index.parquet</code><br />
          <code>results/predictions/predictions.parquet</code><br />
          <code>results/uncertainty/uncertainty.parquet</code><br />
          <code>results/benchmarks/benchmark_results.parquet</code>
        </div>

        <p style={{ fontSize: "0.875rem", color: "var(--color-text-muted)" }}>
          Planned endpoints in Phase 6: <code>GET /api/exports</code> with file hash,
          size validation, and safe download handlers restricted to <code>reports/</code> and <code>results/</code>.
        </p>
      </div>
    </div>
  );
};
