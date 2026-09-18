const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export default function ThreatTrendChart() {
  return (
    <div className="flex flex-1 flex-col gap-[18px] rounded-2xl border border-border bg-surface px-7 py-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[15px] font-bold text-text">Threat Detection Over Time</div>
          <div className="mt-0.5 text-xs text-text-muted">Past 7 days · detections per day</div>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="flex items-center gap-1.5 text-[11px] text-accent-light">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-light" />
            Active trend
          </span>
          <span className="rounded-lg bg-info-dim px-2.5 py-1 text-[11px] font-bold text-info">7D</span>
        </div>
      </div>
      <svg viewBox="0 0 700 200" className="h-[200px] w-full">
        <defs>
          <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-accent)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="var(--color-accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path
          d="M20,150 C80,140 100,110 160,100 C220,90 260,150 320,130 C380,110 420,60 480,55 C540,50 580,90 620,40 L620,190 L20,190 Z"
          fill="url(#areaFill)"
        />
        <path
          d="M20,150 C80,140 100,110 160,100 C220,90 260,150 320,130 C380,110 420,60 480,55 C540,50 580,90 620,40"
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth="3"
          strokeLinecap="round"
        />
        {[
          [20, 150],
          [160, 100],
          [320, 130],
          [480, 55],
        ].map(([cx, cy]) => (
          <circle key={cx} cx={cx} cy={cy} r="4" fill="var(--color-bg)" stroke="var(--color-accent)" strokeWidth="2" />
        ))}
        <circle cx="620" cy="40" r="5" fill="var(--color-accent)" />
      </svg>
      <div className="flex justify-between px-1 font-mono text-[11px] text-text-muted">
        {days.map((d) => (
          <span key={d}>{d}</span>
        ))}
      </div>
    </div>
  );
}
