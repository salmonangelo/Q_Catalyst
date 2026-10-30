import React from "react";
import { SystemStatusResponse } from "../types";

interface Props {
  status: SystemStatusResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}

export const TopStatusBar: React.FC<Props> = ({
  status,
  loading,
  error,
  onRefresh,
}) => {
  const isOnline = !error && status?.status === "ok";

  return (
    <header className="top-status-bar">
      <div className="status-left">
        <div className="connection-pill">
          <span
            className={`status-dot ${
              loading ? "checking" : isOnline ? "online" : "offline"
            }`}
          />
          <span>{loading ? "Checking API..." : isOnline ? "Backend Online" : "Disconnected"}</span>
        </div>

        {status && (
          <>
            <div className="env-tag">
              <span>🐍 Python {status.environment.python_version}</span>
            </div>
            <div className="env-tag">
              <span>⚙️ {status.environment.device.toUpperCase()}</span>
            </div>
            {status.raw_data_available && (
              <div className="env-tag">
                <span>📁 Raw Data Ready</span>
              </div>
            )}
          </>
        )}
      </div>

      <div className="status-right">
        {error && <span style={{ color: "var(--accent-rose)", fontSize: "11px" }}>{error}</span>}
        <button
          className="btn btn-secondary"
          onClick={onRefresh}
          disabled={loading}
          style={{ padding: "4px 10px", fontSize: "11px" }}
        >
          {loading ? <span className="spinner" /> : "↻ Refresh"}
        </button>
      </div>
    </header>
  );
};
