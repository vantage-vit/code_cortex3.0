import Header from "../components/Header";
import StatCard from "../components/StatCard";
import { VerdictBadge, RiskLabel } from "../components/Badge";
import { historyEntries, historyStats } from "../data/mockData";

export default function HistoryPage() {
  return (
    <div className="min-h-screen bg-bg">
      <Header crumb="Analysis History" statusDot />

      <main className="mx-auto max-w-5xl px-6 py-10">
        <div className="flex flex-col gap-4 sm:flex-row">
          <StatCard label="Total Analyses" value={historyStats.totalAnalyses.toLocaleString()} />
          <StatCard label="Malicious Detected" value={historyStats.maliciousDetected} tone="danger" />
          <StatCard label="Benign" value={historyStats.benign} tone="accent" />
        </div>

        <div className="mt-6 overflow-hidden rounded-xl border border-border bg-surface">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-border text-xs tracking-wide text-text-muted">
                <th className="px-6 py-3 font-medium">FILE</th>
                <th className="px-6 py-3 font-medium">DETECTION</th>
                <th className="px-6 py-3 font-medium">CONFIDENCE</th>
                <th className="px-6 py-3 font-medium">RISK</th>
                <th className="px-6 py-3 font-medium">TIME</th>
              </tr>
            </thead>
            <tbody>
              {historyEntries.map((entry) => (
                <tr key={entry.fileName + entry.time} className="border-b border-border last:border-none">
                  <td className="px-6 py-4 text-text">{entry.fileName}</td>
                  <td className="px-6 py-4">
                    <VerdictBadge verdict={entry.verdict} />
                  </td>
                  <td className="px-6 py-4 text-text">{entry.confidence.toFixed(1)}%</td>
                  <td className="px-6 py-4">
                    <RiskLabel risk={entry.risk} />
                  </td>
                  <td className="px-6 py-4 text-text-muted">{entry.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
