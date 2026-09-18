import { Link } from "react-router-dom";
import type { RiskLevel } from "../../types";
import { historyEntries } from "../../data/mockData";

const riskDotColor: Record<RiskLevel, string> = {
  CRITICAL: "var(--color-danger)",
  HIGH: "var(--color-warning)",
  MEDIUM: "var(--color-gold)",
  LOW: "var(--color-accent-light)",
};

const riskTextColor: Record<RiskLevel, string> = {
  CRITICAL: "text-danger",
  HIGH: "text-warning",
  MEDIUM: "text-gold",
  LOW: "text-accent-light",
};

export default function RecentDetections() {
  return (
    <div className="flex flex-col gap-[18px] rounded-2xl border border-border bg-surface px-7 py-6">
      <div className="flex flex-wrap items-center justify-between gap-2.5">
        <div>
          <div className="text-[15px] font-bold text-text">Recent Detections</div>
          <div className="mt-0.5 text-xs text-text-muted">History of analyses from the last hour</div>
        </div>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5 rounded-lg bg-accent-dim px-2.5 py-1 text-[11px] font-bold text-accent-light">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-light" />
            LIVE FEED
          </span>
          <Link to="/history" className="text-[11px] text-text-muted hover:text-text">
            View all →
          </Link>
        </div>
      </div>
      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-border">
            <th className="pb-2.5 text-left text-[11px] font-semibold tracking-wide text-text-muted">FILE</th>
            <th className="pb-2.5 text-left text-[11px] font-semibold tracking-wide text-text-muted">STATUS</th>
            <th className="pb-2.5 text-left text-[11px] font-semibold tracking-wide text-text-muted">CONFIDENCE</th>
            <th className="pb-2.5 text-left text-[11px] font-semibold tracking-wide text-text-muted">RISK</th>
            <th className="pb-2.5 text-left text-[11px] font-semibold tracking-wide text-text-muted">TIME</th>
            <th className="pb-2.5 text-left text-[11px] font-semibold tracking-wide text-text-muted">ACTION</th>
          </tr>
        </thead>
        <tbody>
          {historyEntries.map((entry, i) => (
            <tr
              key={entry.fileName + entry.time}
              className={i < historyEntries.length - 1 ? "border-b border-border-strong/60" : ""}
            >
              <td className="py-3 pr-3">
                <Link
                  to={`/results/${encodeURIComponent(entry.fileName)}`}
                  className="flex items-center gap-2 text-text hover:text-accent-light"
                >
                  <span className="h-[7px] w-[7px] rounded-full" style={{ background: riskDotColor[entry.risk] }} />
                  {entry.fileName}
                </Link>
              </td>
              <td className="px-3 py-3">
                <span className={`text-[11px] font-bold ${entry.verdict === "MALICIOUS" ? "text-danger" : "text-accent-light"}`}>
                  {entry.verdict}
                </span>
              </td>
              <td className="px-3 py-3 font-mono text-text">{entry.confidence.toFixed(1)}%</td>
              <td className={`px-3 py-3 font-semibold ${riskTextColor[entry.risk]}`}>{entry.risk}</td>
              <td className="px-3 py-3 font-mono text-text-muted">{entry.time}</td>
              <td className="py-3 pl-3 text-text-muted">{entry.action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
