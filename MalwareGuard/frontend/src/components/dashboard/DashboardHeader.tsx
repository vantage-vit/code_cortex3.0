import { Link } from "react-router-dom";

export default function DashboardHeader() {
  return (
    <header className="flex items-center gap-3.5 border-b border-border bg-surface px-10 py-[18px]">
      <span className="h-2.5 w-2.5 rounded-full bg-accent shadow-[0_0_8px_#39c99a99]" />
      <Link to="/" className="text-[19px] font-bold tracking-tight text-text">
        MalwareGuard
      </Link>
      <span className="text-[13px] text-text-muted">Analysis Dashboard</span>
      <span className="flex-1" />
      <Link
        to="/history"
        className="hidden text-[13px] text-text-muted hover:text-text sm:inline"
      >
        History
      </Link>
      <span className="flex items-center gap-1.5 rounded-full border border-accent-dim bg-accent-dim px-3 py-1.5 font-mono text-[11px] font-semibold tracking-wide text-accent-light">
        <span className="h-1.5 w-1.5 rounded-full bg-accent-light" />
        LIVE
      </span>
    </header>
  );
}
