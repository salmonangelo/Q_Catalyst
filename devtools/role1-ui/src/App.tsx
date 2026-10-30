import React, { useState, useEffect, useCallback } from "react";
import { Sidebar } from "./components/Sidebar";
import { TopStatusBar } from "./components/TopStatusBar";
import { fetchSystemStatus } from "./lib/api";
import { SystemStatusResponse, StageStatus } from "./types";

import { DataAuditPage } from "./pages/DataAuditPage";
import { MutationInspectorPage } from "./pages/MutationInspectorPage";
import { Esm1vPage } from "./pages/Esm1vPage";
import { Esm2Page } from "./pages/Esm2Page";
import { XGBoostPage } from "./pages/XGBoostPage";
import { UncertaintyPage } from "./pages/UncertaintyPage";
import { Gate0Page } from "./pages/Gate0Page";
import { ExportsPage } from "./pages/ExportsPage";

const DEFAULT_STAGES: StageStatus[] = [
  { stage_id: 1, name: "Data Audit & Cleaning", status: "ready", description: "Audit raw PET-Gym and PAZy files", implemented: true },
  { stage_id: 2, name: "Mutation Inspector", status: "ready", description: "Validate notation and sequence offsets", implemented: true },
  { stage_id: 3, name: "ESM-1v Scoring", status: "ready", description: "Zero-shot masked-marginal effect scoring", implemented: true },
  { stage_id: 4, name: "ESM-2 Embeddings", status: "ready", description: "Per-sequence embedding extraction and caching", implemented: true },
  { stage_id: 5, name: "XGBoost Predictor", status: "ready", description: "Fit XGBoost and Ridge with held-out splits", implemented: true },
  { stage_id: 6, name: "Uncertainty Estimation", status: "ready", description: "Bootstrap ensemble and conformal intervals", implemented: true },
  { stage_id: 7, name: "Gate 0 & Benchmarks", status: "ready", description: "Uncertainty error correlation and A0-A2 baselines", implemented: true },
  { stage_id: 8, name: "Reports & Exports", status: "ready", description: "Artifact downloads and audit summaries", implemented: true },
];

export const App: React.FC = () => {
  const [activeStageId, setActiveStageId] = useState<number>(1);
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState<boolean>(true);
  const [statusError, setStatusError] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    setStatusLoading(true);
    setStatusError(null);
    try {
      const data = await fetchSystemStatus();
      setSystemStatus(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to connect to backend";
      setStatusError(msg);
    } finally {
      setStatusLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  const stages = systemStatus?.stages && systemStatus.stages.length > 0
    ? systemStatus.stages
    : DEFAULT_STAGES;

  const renderActivePage = () => {
    switch (activeStageId) {
      case 1:
        return <DataAuditPage />;
      case 2:
        return <MutationInspectorPage />;
      case 3:
        return <Esm1vPage />;
      case 4:
        return <Esm2Page />;
      case 5:
        return <XGBoostPage />;
      case 6:
        return <UncertaintyPage />;
      case 7:
        return <Gate0Page />;
      case 8:
        return <ExportsPage />;
      default:
        return <DataAuditPage />;
    }
  };

  return (
    <div className="app-shell">
      <Sidebar
        activeStageId={activeStageId}
        onSelectStage={(id) => setActiveStageId(id)}
        stages={stages}
      />
      <div className="main-content">
        <TopStatusBar
          status={systemStatus}
          loading={statusLoading}
          error={statusError}
          onRefresh={loadStatus}
        />
        <main className="page-body">
          {renderActivePage()}
        </main>
      </div>
    </div>
  );
};
