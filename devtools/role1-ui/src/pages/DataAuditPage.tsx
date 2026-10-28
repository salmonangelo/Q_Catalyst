import React, { useEffect, useState } from "react";
import { fetchAuditData, getDownloadUrl, runDataAudit } from "../lib/api";
import { AuditDataResponse, RunAuditResponse } from "../types";
import { MetricCard } from "../components/MetricCard";

export const DataAuditPage: React.FC = () => {
  const [auditData, setAuditData] = useState<AuditDataResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [runningAudit, setRunningAudit] = useState<boolean>(false);
  const [runResult, setRunResult] = useState<RunAuditResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"table" | "report">("table");
  const [searchTerm, setSearchTerm] = useState<string>("");

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchAuditData();
      setAuditData(data);
    } catch (err: any) {
      setError(err?.message || "Failed to load audit data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleRunAudit = async () => {
    try {
      setRunningAudit(true);
      setError(null);
      const res = await runDataAudit();
      setRunResult(res);
      await loadData();
    } catch (err: any) {
      setError(err?.message || "Audit execution failed");
    } finally {
      setRunningAudit(false);
    }
  };

  const filteredCsvs = (auditData?.csv_audits || []).filter((item) => {
    const q = searchTerm.toLowerCase();
    return (
      item.file_name.toLowerCase().includes(q) ||
      item.enzyme.toLowerCase().includes(q) ||
      item.dataset_type.toLowerCase().includes(q) ||
      item.anomaly_flags.toLowerCase().includes(q)
    );
  });

  return (
    <div className="workspace-container">
      <div className="page-header">
        <div className="page-title-row">
          <h2 className="page-title">Stage 1: Raw Data Audit & Cleaning</h2>
          <div className="btn-group">
            <button
              className="btn btn-secondary"
              onClick={loadData}
              disabled={loading || runningAudit}
            >
              ↻ Refresh
            </button>
            <button
              className="btn btn-primary"
              onClick={handleRunAudit}
              disabled={runningAudit}
            >
              {runningAudit ? (
                <>
                  <span className="spinner" /> Auditing Raw Files...
                </>
              ) : (
                "▶ Run Data Audit"
              )}
            </button>
          </div>
        </div>
        <p className="page-description">
          Read-only, non-destructive audit of raw PET-Gym and PAZy experimental datasets.
          Inventories files, calculates cryptographic SHA-256 hashes, validates CSV headers
          without assuming schema, and identifies duplicates and unphysical outliers.
        </p>
      </div>

      {/* Directory Paths Info */}
      <div className="info-box">
        <strong>Source Paths:</strong> PET-Gym at{" "}
        <span className="path-badge">data/raw/petflow/petflow/PET-Gym</span> · PAZy at{" "}
        <span className="path-badge">data/raw/pazy</span> · Output Directory:{" "}
        <span className="path-badge">reports/tables/</span>
      </div>

      {error && (
        <div className="info-box" style={{ borderLeftColor: "var(--accent-rose)", color: "#fecaca" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {runResult && (
        <div className="info-box" style={{ borderLeftColor: "var(--accent-emerald)", color: "#d1fae5" }}>
          <strong>Success:</strong> {runResult.message} (Scanned {runResult.total_files_scanned} files,{" "}
          {runResult.total_csvs_audited} CSVs in {runResult.runtime_seconds}s)
        </div>
      )}

      {/* Key Metrics Grid */}
      <div className="metrics-grid">
        <MetricCard
          label="Total Discovered Files"
          value={auditData?.total_files ?? "—"}
          subtitle="FASTA, CSV, PDB, and text files"
          icon="📂"
          accent="blue"
        />
        <MetricCard
          label="Audited CSV Files"
          value={auditData?.total_csv_files ?? "—"}
          subtitle="Experimental activity & Tm assays"
          icon="📊"
          accent="cyan"
        />
        <MetricCard
          label="Measurement Records"
          value={auditData?.total_records ? auditData.total_records.toLocaleString() : "—"}
          subtitle="Total variant measurement rows"
          icon="🧪"
          accent="emerald"
        />
        <MetricCard
          label="Audit Status"
          value={auditData?.status === "completed" ? "Verified" : "Not Run"}
          subtitle="100% local inspection, read-only"
          icon="🛡️"
          accent={auditData?.status === "completed" ? "emerald" : "amber"}
        />
      </div>

      {/* Actions and Downloads Toolbar */}
      <div className="toolbar">
        <div className="btn-group">
          <a
            href={getDownloadUrl("petgym_file_manifest.csv")}
            className="btn btn-download"
            download
          >
            ⬇ Download Manifest CSV
          </a>
          <a
            href={getDownloadUrl("petgym_csv_audit.csv")}
            className="btn btn-download"
            download
          >
            ⬇ Download CSV Audit Table
          </a>
          <a
            href={getDownloadUrl("data_audit_summary.md")}
            className="btn btn-download"
            download
          >
            ⬇ Download Markdown Report
          </a>
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === "table" ? "active" : ""}`}
          onClick={() => setActiveTab("table")}
        >
          CSV Dataset Audits ({auditData?.csv_audits.length || 0})
        </button>
        <button
          className={`tab-btn ${activeTab === "report" ? "active" : ""}`}
          onClick={() => setActiveTab("report")}
        >
          Readable Audit Report
        </button>
      </div>

      {/* Tab 1: CSV Audit Table */}
      {activeTab === "table" && (
        <div className="table-card">
          <div className="table-header-controls">
            <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-secondary)" }}>
              Audited CSV Files & Quality Diagnostics
            </span>
            <input
              type="text"
              className="search-input"
              placeholder="Search by enzyme, split, or anomaly..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>File Name</th>
                  <th>Enzyme</th>
                  <th>Split Type</th>
                  <th>Target</th>
                  <th>Rows</th>
                  <th>Cols</th>
                  <th>Missing</th>
                  <th>Dup Rows</th>
                  <th>Dup Mutants</th>
                  <th>Score Range</th>
                  <th>Anomaly Flags</th>
                </tr>
              </thead>
              <tbody>
                {filteredCsvs.length === 0 ? (
                  <tr>
                    <td colSpan={11} style={{ textAlign: "center", padding: "24px", color: "var(--text-muted)" }}>
                      {loading ? "Loading audit data..." : "No CSV datasets found matching filter."}
                    </td>
                  </tr>
                ) : (
                  filteredCsvs.map((row, idx) => (
                    <tr key={idx}>
                      <td style={{ fontFamily: "var(--font-mono)", fontWeight: 500 }}>{row.file_name}</td>
                      <td>
                        <span style={{ color: "#38bdf8", fontWeight: 600 }}>{row.enzyme}</span>
                      </td>
                      <td>{row.dataset_type}</td>
                      <td>
                        <span
                          className={`badge ${
                            row.target_type === "Act"
                              ? "badge-completed"
                              : row.target_type === "Tm"
                              ? "badge-ready"
                              : "badge-not_implemented"
                          }`}
                        >
                          {row.target_type}
                        </span>
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{row.row_count}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{row.col_count}</td>
                      <td style={{ fontFamily: "var(--font-mono)", color: row.missing_cells > 0 ? "var(--accent-amber)" : "inherit" }}>
                        {row.missing_cells}
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)", color: row.duplicate_rows > 0 ? "var(--accent-rose)" : "inherit" }}>
                        {row.duplicate_rows}
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)", color: row.duplicate_mutants > 0 ? "var(--accent-rose)" : "inherit" }}>
                        {row.duplicate_mutants}
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px" }}>
                        {row.min_score !== null ? `[${row.min_score}, ${row.max_score}]` : "N/A"}
                      </td>
                      <td>
                        {row.anomaly_flags !== "none" ? (
                          <span className="badge badge-failed" style={{ textTransform: "none" }}>
                            {row.anomaly_flags}
                          </span>
                        ) : (
                          <span style={{ color: "var(--text-muted)", fontSize: "11px" }}>clean</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Readable Audit Markdown Report */}
      {activeTab === "report" && (
        <div className="markdown-container">
          {auditData?.summary_markdown || "No summary report text available."}
        </div>
      )}
    </div>
  );
};
