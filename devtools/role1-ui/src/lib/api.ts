import { AuditDataResponse, RunAuditResponse, SystemStatusResponse } from "../types";

const API_BASE = ""; // Relative path leveraging Vite proxy to /api

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch system status: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchAuditData(): Promise<AuditDataResponse> {
  const res = await fetch(`${API_BASE}/api/audit`);
  if (!res.ok) {
    throw new Error(`Failed to fetch audit data: ${res.statusText}`);
  }
  return res.json();
}

export async function runDataAudit(): Promise<RunAuditResponse> {
  const res = await fetch(`${API_BASE}/api/audit/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => null);
    throw new Error(errorData?.detail || `Audit execution failed: ${res.statusText}`);
  }
  return res.json();
}

export function getDownloadUrl(filename: string): string {
  return `${API_BASE}/api/audit/download/${encodeURIComponent(filename)}`;
}
