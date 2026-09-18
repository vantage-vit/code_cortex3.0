import type { RiskLevel, Verdict } from "../types";

const verdictStyles: Record<Verdict, string> = {
  MALICIOUS: "bg-danger-dim text-danger",
  BENIGN: "bg-accent-dim text-accent",
};

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <span
      className={`inline-flex items-center rounded px-2 py-1 text-xs font-semibold tracking-wide ${verdictStyles[verdict]}`}
    >
      {verdict}
    </span>
  );
}

const riskStyles: Record<RiskLevel, string> = {
  CRITICAL: "text-danger",
  HIGH: "text-danger",
  MEDIUM: "text-warning",
  LOW: "text-accent",
};

export function RiskLabel({ risk }: { risk: RiskLevel }) {
  return <span className={`text-xs font-semibold ${riskStyles[risk]}`}>{risk}</span>;
}

export function RiskPill({ risk }: { risk: RiskLevel }) {
  const critical = risk === "CRITICAL" || risk === "HIGH";
  return (
    <span
      className={`inline-flex items-center rounded px-3 py-1 text-xs font-semibold tracking-wide ${
        critical ? "bg-danger-dim text-danger" : "bg-accent-dim text-accent"
      }`}
    >
      {risk} RISK
    </span>
  );
}
