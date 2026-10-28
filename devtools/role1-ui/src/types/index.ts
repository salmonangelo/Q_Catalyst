export interface EnvironmentInfo {
  python_version: string;
  platform: string;
  torch_version: string | null;
  transformers_version: string | null;
  xgboost_version: string | null;
  sklearn_version: string | null;
  device: string;
}

export type ProcessStatus = "not_implemented" | "ready" | "running" | "completed" | "failed";

export interface StageStatus {
  stage_id: number;
  name: string;
  status: ProcessStatus;
  description: string;
  implemented: boolean;
}

export interface SystemStatusResponse {
  status: string;
  environment: EnvironmentInfo;
  raw_data_available: boolean;
  curated_data_available: boolean;
  predictions_available: boolean;
  uncertainty_available: boolean;
  gate0_available: boolean;
  benchmarks_available: boolean;
  stages: StageStatus[];
}

export interface AuditCsvSummary {
  file_path: string;
  file_name: string;
  dataset_type: string;
  enzyme: string;
  target_type: string;
  status: string;
  row_count: number;
  col_count: number;
  columns: string;
  missing_cells: number;
  duplicate_rows: number;
  duplicate_mutants: number;
  min_score: number | null;
  max_score: number | null;
  anomaly_flags: string;
  error_message: string;
}

export interface AuditDataResponse {
  status: "completed" | "not_run";
  raw_dir: string;
  total_files: number;
  total_csv_files: number;
  total_records: number;
  manifest_file: string;
  manifest_file_size: number;
  audit_file: string;
  audit_file_size: number;
  summary_file: string;
  summary_file_size: number;
  csv_audits: AuditCsvSummary[];
  summary_markdown: string;
}

export interface RunAuditResponse {
  status: string;
  runtime_seconds: number;
  total_files_scanned: number;
  total_csvs_audited: number;
  total_measurement_records: number;
  manifest_path: string;
  manifest_size_bytes: number;
  audit_path: string;
  audit_size_bytes: number;
  summary_path: string;
  summary_size_bytes: number;
  message: string;
}
