import React from "react";

export const Esm2Page: React.FC = () => {
  return (
    <div className="workspace-container">
      <div className="page-header">
        <div className="page-title-row">
          <h2 className="page-title">Stage 4: ESM-2 Embedding Extraction</h2>
          <span className="badge badge-ready">Phase 3 Scheduled</span>
        </div>
        <p className="page-description">
          Extract 480-dimensional mean-pooled sequence representations with sequence-hash cache indexing.
        </p>
      </div>

      <div className="stub-card">
        <div className="stub-icon">🧬</div>
        <h3 className="stub-title">ESM-2 Embeddings Interface Scheduled for Phase 3</h3>
        <p className="stub-desc">
          Embedding pipeline in <span className="path-badge">src/qcatalyst/protein_ai/embeddings.py</span> is
          implemented with SHA-256 `.npz` caching and Parquet index tracking.
        </p>

        <div className="stub-details">
          <h4>Engine Capabilities</h4>
          <ul>
            <li>
              <span className="func">ESM2Embedder(config)</span> — PyTorch forward evaluation with zero gradients
            </li>
            <li>
              <span className="func">extract_batch(df, pooling="mean")</span> — Deterministic batch vectorization
            </li>
            <li>
              <span className="func">results/embeddings/embedding_index.parquet</span> — Cache provenance index
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
