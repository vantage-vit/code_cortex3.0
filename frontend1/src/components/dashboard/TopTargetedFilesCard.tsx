import { topTargetedFiles } from "../../data/mockData";

export default function TopTargetedFilesCard() {
  return (
    <div className="flex flex-1 flex-col gap-[18px] rounded-2xl border border-border bg-surface px-7 py-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[15px] font-bold text-text">Top Targeted Files</div>
          <div className="mt-0.5 text-xs text-text-muted">Highest-confidence detections in the last 24 hours</div>
        </div>
        <span className="rounded-lg bg-accent-dim px-2.5 py-1 text-[11px] font-bold text-accent-light">LIVE</span>
      </div>
      <div className="flex flex-col gap-3.5">
        {topTargetedFiles.map((file) => (
          <div key={file.fileName} className="flex flex-col gap-1.5">
            <div className="flex justify-between text-[13px]">
              <span className="text-text">{file.fileName}</span>
              <span className="font-mono text-text-muted">{file.confidence.toFixed(1)}%</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
              <div
                className="h-full rounded-full"
                style={{ width: `${file.confidence}%`, background: file.color }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
