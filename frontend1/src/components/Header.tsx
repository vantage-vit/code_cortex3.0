import { Link } from "react-router-dom";

interface HeaderProps {
  crumb?: string;
  statusDot?: boolean;
}

export default function Header({ crumb, statusDot }: HeaderProps) {
  return (
    <header className="flex items-center gap-3 border-b border-border bg-surface px-6 py-4">
      {statusDot ? (
        <span className="h-2 w-2 rounded-full bg-accent" />
      ) : (
        <span className="h-4 w-4 rounded-full bg-accent" />
      )}
      <Link to="/" className="text-base font-bold tracking-tight text-text">
        MalwareGuard
      </Link>
      {crumb && (
        <>
          <span className="text-text-subtle">·</span>
          <span className="text-sm text-text-muted">{crumb}</span>
        </>
      )}
      <Link to="/history" className="ml-auto text-sm text-text-muted hover:text-text">
        History
      </Link>
    </header>
  );
}
