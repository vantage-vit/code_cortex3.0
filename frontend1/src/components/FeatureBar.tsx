import type { FeatureContribution } from "../types";

export default function FeatureBar({ name, impact }: FeatureContribution) {
  const width = Math.min(Math.abs(impact) / 0.35, 1) * 100;

  return (
    <div className="py-2.5">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-text">{name}</span>
        <span className="text-sm font-semibold text-danger">+{impact.toFixed(2)}</span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
        <div className="h-full rounded-full bg-danger" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}
