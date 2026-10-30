import React from "react";
import { StageStatus } from "../types";
import { ProcessBadge } from "./ProcessBadge";

interface Props {
  activeStageId: number;
  onSelectStage: (id: number) => void;
  stages: StageStatus[];
}

export const Sidebar: React.FC<Props> = ({
  activeStageId,
  onSelectStage,
  stages,
}) => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon">🧬</div>
        <div className="brand-info">
          <h1>Q-Catalyst</h1>
          <span>Role 1 · Protein ML</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {stages.map((stage) => {
          const isActive = stage.stage_id === activeStageId;
          return (
            <button
              key={stage.stage_id}
              className={`nav-item ${isActive ? "active" : ""}`}
              onClick={() => onSelectStage(stage.stage_id)}
            >
              <div className="nav-left">
                <span className="nav-number">{stage.stage_id}</span>
                <span className="nav-label">{stage.name}</span>
              </div>
              <ProcessBadge status={stage.status} />
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <span>PETase AI Workbench</span>
        <span>v1.0.0</span>
      </div>
    </aside>
  );
};
