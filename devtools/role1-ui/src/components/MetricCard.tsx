import React from "react";

interface Props {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: string;
  accent?: "blue" | "emerald" | "amber" | "rose" | "purple" | "cyan";
}

export const MetricCard: React.FC<Props> = ({
  label,
  value,
  subtitle,
  icon,
  accent = "blue",
}) => {
  return (
    <div className={`metric-card metric-card-${accent}`}>
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        {icon && <span className="metric-icon">{icon}</span>}
      </div>
      <div className="metric-value">{value}</div>
      {subtitle && <div className="metric-subtitle">{subtitle}</div>}
    </div>
  );
};
