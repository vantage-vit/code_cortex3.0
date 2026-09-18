import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { sampleFiles } from "../../data/mockData";

export default function UploadPanel() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleAnalyze = () => {
    if (selectedFile) {
      navigate(`/results/${encodeURIComponent(selectedFile)}`);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) setSelectedFile(file.name);
  };

  return (
    <div className="flex flex-col gap-[18px] rounded-2xl border border-border bg-surface px-8 py-7">
      <div>
        <div className="text-[17px] font-bold text-text">Analyze a New File</div>
        <div className="mt-0.5 text-[13px] text-text-muted">
          Upload a PE feature sample or try one of the labeled test files
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-6">
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`flex min-w-[320px] flex-1 cursor-pointer flex-col items-center gap-2.5 rounded-xl border-2 border-dashed bg-surface-3 px-8 py-8 transition-colors ${
            isDragging ? "border-accent-light" : "border-accent-dim"
          }`}
        >
          <span className="flex h-[42px] w-[42px] items-center justify-center rounded-full bg-accent-dim">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent-light">
              <path d="M12 16V4" />
              <path d="m6 10 6-6 6 6" />
              <path d="M4 20h16" />
            </svg>
          </span>
          <span className="text-sm font-semibold text-text">
            {selectedFile ? selectedFile : "Drop a feature file or click to browse"}
          </span>
          <span className="text-xs text-text-muted">Supports .csv, .json</span>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.json"
            className="absolute h-px w-px overflow-hidden opacity-0"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) setSelectedFile(file.name);
            }}
          />
        </label>
        <div className="flex min-w-[220px] flex-col gap-3">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile}
            className="rounded-lg bg-accent px-7 py-3 text-[13px] font-bold tracking-wide text-bg transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            ANALYZE
          </button>
          <div className="flex flex-wrap gap-2">
            {sampleFiles.map((file) => (
              <button
                key={file}
                onClick={() => setSelectedFile(file)}
                className={`rounded-full border px-3.5 py-1.5 text-xs transition-colors ${
                  selectedFile === file
                    ? "border-accent bg-accent text-bg font-semibold"
                    : "border-border bg-surface-2 text-text hover:border-border-strong"
                }`}
              >
                {file}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
