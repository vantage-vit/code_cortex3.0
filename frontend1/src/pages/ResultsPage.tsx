import { Link, useParams } from "react-router-dom";
import Header from "../components/Header";
import { RiskPill } from "../components/Badge";
import FeatureBar from "../components/FeatureBar";
import { mockAnalysisResults } from "../data/mockData";

const featureLabels: Record<string, string> = {
  sections: "Sections",
  dllImports: "DLL Imports",
  totalImports: "Total Imports",
  resources: "Resources",
  maxSectionEntropy: "Max Section Entropy",
};

export default function ResultsPage() {
  const { fileName = "" } = useParams();
  const result = mockAnalysisResults[decodeURIComponent(fileName)];

  if (!result) {
    return (
      <div className="min-h-screen bg-bg">
        <Header crumb="Not found" />
        <main className="mx-auto max-w-2xl px-6 py-24 text-center">
          <p className="text-text-muted">No analysis found for "{fileName}".</p>
          <Link to="/" className="mt-4 inline-block text-accent hover:underline">
            Back to analyze
          </Link>
        </main>
      </div>
    );
  }

  const isMalicious = result.verdict === "MALICIOUS";

  return (
    <div className="min-h-screen bg-bg">
      <Header crumb={result.fileName} />

      <main className="mx-auto grid max-w-5xl grid-cols-1 gap-6 px-6 py-10 md:grid-cols-2">
        <div className="flex flex-col gap-6">
          <div className={`rounded-xl border p-6 ${isMalicious ? "border-danger/40" : "border-accent/40"}`}>
            <RiskPill risk={result.risk} />
            <h1 className={`mt-4 text-3xl font-bold ${isMalicious ? "text-danger" : "text-accent"}`}>
              {result.verdict}
            </h1>
            <p className="mt-1 text-text-muted">{result.confidence.toFixed(1)}% confidence</p>
            <hr className="my-4 border-border" />
            <p className="font-mono text-xs text-text-subtle">MD5: {result.md5}</p>
          </div>

          <div className="rounded-xl border border-border bg-surface p-6">
            <h2 className="text-sm font-semibold tracking-wide text-text-muted">PE FEATURES</h2>
            <dl className="mt-4 flex flex-col gap-3">
              {Object.entries(result.features).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between border-b border-border/60 pb-2 last:border-none last:pb-0">
                  <dt className="text-sm text-text-muted">{featureLabels[key] ?? key}</dt>
                  <dd className="text-sm font-semibold text-text">{value}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        <div className="flex flex-col justify-between rounded-xl border border-border bg-surface p-6">
          <div>
            <h2 className="text-lg font-semibold text-text">Why was this flagged?</h2>
            <p className="mt-1 text-sm text-text-muted">Top contributing features, by impact on the prediction</p>

            <div className="mt-4 flex flex-col divide-y divide-border">
              {result.topFeatures.map((f) => (
                <FeatureBar key={f.name} name={f.name} impact={f.impact} />
              ))}
            </div>
          </div>

          <button className="mt-6 w-fit rounded-md border border-border-strong bg-surface-2 px-5 py-2.5 text-sm font-medium text-text hover:bg-border-strong">
            Generate Threat Report
          </button>
        </div>
      </main>
    </div>
  );
}
