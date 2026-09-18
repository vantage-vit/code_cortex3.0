import { responseStats, riskBreakdown } from "../../data/mockData";

export default function RiskBreakdownCard() {
  const total = riskBreakdown.reduce((sum, r) => sum + r.count, 0);

  return (
    <div className="flex w-full max-w-[360px] flex-col gap-[18px] rounded-2xl border border-border bg-surface px-7 py-6">
      <div>
        <div className="text-[15px] font-bold text-text">Risk Level Breakdown</div>
        <div className="mt-0.5 text-xs text-text-muted">Current status distribution</div>
      </div>
      <div className="flex h-2.5 overflow-hidden rounded-full">
        {riskBreakdown.map((r) => (
          <span key={r.level} style={{ width: `${(r.count / total) * 100}%`, background: r.color }} />
        ))}
      </div>
      <div className="flex flex-col gap-2">
        {riskBreakdown.map((r) => (
          <div key={r.level} className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />
              {r.level.charAt(0) + r.level.slice(1).toLowerCase()}
            </span>
            <span className="font-mono text-text-muted">{r.count}</span>
          </div>
        ))}
      </div>
      <div className="mt-1 flex gap-3">
        <div className="flex-1 rounded-xl border border-border bg-surface-3 px-3.5 py-3">
          <div className="text-[10px] text-text-muted">Avg Response</div>
          <div className="mt-0.5 font-mono text-[15px] font-bold text-text">{responseStats.avgResponse}</div>
        </div>
        <div className="flex-1 rounded-xl border border-border bg-surface-3 px-3.5 py-3">
          <div className="text-[10px] text-text-muted">Escalations</div>
          <div className="mt-0.5 font-mono text-[15px] font-bold text-text">{responseStats.escalations}</div>
        </div>
      </div>
    </div>
  );
}
