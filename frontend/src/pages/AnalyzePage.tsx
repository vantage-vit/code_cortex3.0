import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import { sampleFiles } from "../data/mockData";

export default function AnalyzePage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleAnalyze = () => {
    if (selectedFile) {
      navigate(`/results/${encodeURIComponent(selectedFile)}`);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) setSelectedFile(file.name);
  };

  return (
    <div className="min-h-screen bg-bg">
      <Header crumb="Static PE Malware Detection" />

      <main className="mx-auto flex max-w-2xl flex-col items-center px-6 py-24 text-center">
        <h1 className="text-4xl font-bold text-text">Analyze a File</h1>
        <p className="mt-3 text-text-muted">Upload a PE feature sample or pick one from the test set</p>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`mt-10 flex w-full cursor-pointer flex-col items-center rounded-xl border-2 border-dashed px-8 py-14 transition-colors ${
            isDragging ? "border-accent bg-surface-2" : "border-accent/40 bg-surface"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.json"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) setSelectedFile(file.name);
            }}
          />
          <div className="h-14 w-14 rounded-full bg-surface-2" />
          <p className="mt-5 text-lg font-semibold text-text">
            {selectedFile ? selectedFile : "Drop a feature file or click to browse"}
          </p>
          <p className="mt-2 text-sm text-text-muted">Supports .csv, .json · or pick a labeled sample below</p>

          <button
            onClick={(e) => {
              e.stopPropagation();
              handleAnalyze();
            }}
            disabled={!selectedFile}
            className="mt-6 rounded-md bg-accent px-8 py-2.5 text-sm font-semibold text-bg transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            ANALYZE
          </button>
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-center gap-2 rounded-md border border-border bg-surface px-4 py-2">
          <span className="text-sm text-text-muted">Try a sample:</span>
          {sampleFiles.map((file) => (
            <button
              key={file}
              onClick={() => setSelectedFile(file)}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                selectedFile === file
                  ? "bg-accent text-bg"
                  : "bg-surface-2 text-text hover:bg-border-strong"
              }`}
            >
              {file}
            </button>
          ))}
        </div>
      </main>
    </div>
  );
}
