import React from "react";
import { ProcessStatus } from "../types";

interface Props {
  status: ProcessStatus;
  label?: string;
}

export const ProcessBadge: React.FC<Props> = ({ status, label }) => {
  const getLabel = () => {
    if (label) return label;
    switch (status) {
      case "completed":
        return "Completed";
      case "ready":
        return "Ready";
      case "running":
        return "Running";
      case "failed":
        return "Failed";
      case "not_implemented":
      default:
        return "Not implemented";
    }
  };

  return <span className={`badge badge-${status}`}>{getLabel()}</span>;
};
