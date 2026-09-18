import { threatDistribution } from "../../data/mockData";

export default function ThreatDistribution({ total }: { total: number }) {
  let cursor = 0;
  const stops = threatDistribution.map((slice) => {
    const start = cursor;
    cursor += slice.pct;
    return `${slice.color} ${start}% ${cursor}%`;
  });

  return (
    <div className="flex w-full max-w-[360px] flex-col gap-[18px] rounded-2xl border border-border bg-surface px-7 py-6">
      <div>
        <div className="text-[15px] font-bold text-text">Threat Distribution</div>
        <div className="mt-0.5 text-xs text-text-muted">by type</div>
      </div>
      <div className="flex items-center justify-center">
        <div
          className="flex h-[120px] w-[120px] items-center justify-center rounded-full"
          style={{ background: `conic-gradient(${stops.join(", ")})` }}
        >
          <div className="flex h-[76px] w-[76px] flex-col items-center justify-center rounded-full bg-surface">
            <span className="font-mono text-[19px] font-bold text-text">{total}</span>
            <span className="text-[10px] text-text-muted">threats</span>
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-2">
        {threatDistribution.map((slice) => (
          <div key={slice.label} className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full" style={{ background: slice.color }} />
              {slice.label}
            </span>
            <span className="font-mono text-text-muted">{slice.pct}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
