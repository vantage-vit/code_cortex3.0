interface StatCardProps {
  label: string;
  value: string | number;
  tone?: "default" | "danger" | "accent";
}

const toneStyles: Record<NonNullable<StatCardProps["tone"]>, string> = {
  default: "text-text",
  danger: "text-danger",
  accent: "text-accent",
};

export default function StatCard({ label, value, tone = "default" }: StatCardProps) {
  return (
    <div className="flex-1 rounded-lg border border-border bg-surface px-6 py-5">
      <p className="text-sm text-text-muted">{label}</p>
      <p className={`mt-2 text-3xl font-bold ${toneStyles[tone]}`}>{value}</p>
    </div>
  );
}
