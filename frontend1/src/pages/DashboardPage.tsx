import DashboardHeader from "../components/dashboard/DashboardHeader";
import StatCardIcon from "../components/dashboard/StatCardIcon";
import UploadPanel from "../components/dashboard/UploadPanel";
import ThreatTrendChart from "../components/dashboard/ThreatTrendChart";
import ThreatDistribution from "../components/dashboard/ThreatDistribution";
import TopTargetedFilesCard from "../components/dashboard/TopTargetedFilesCard";
import RiskBreakdownCard from "../components/dashboard/RiskBreakdownCard";
import RecentDetections from "../components/dashboard/RecentDetections";
import { historyStats } from "../data/mockData";

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-bg">
      <DashboardHeader />

      <main className="mx-auto flex max-w-[1320px] flex-col gap-6 px-8 pb-16 pt-7">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCardIcon
            label="Total Analyses"
            value={historyStats.totalAnalyses.toLocaleString()}
            trend={historyStats.totalAnalysesTrend}
            trendUp
            iconBg="var(--color-info-dim)"
            iconColor="var(--color-info)"
            icon={
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2 4 5v6c0 5.2 3.4 9.5 8 11 4.6-1.5 8-5.8 8-11V5l-8-3Z" />
              </svg>
            }
          />
          <StatCardIcon
            label="Malicious Detected"
            value={historyStats.maliciousDetected}
            trend={historyStats.maliciousTrend}
            trendUp
            iconBg="var(--color-warning-dim)"
            iconColor="var(--color-warning)"
            icon={
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 3 2 20h20L12 3Z" />
                <path d="M12 10v4" />
                <path d="M12 17h.01" />
              </svg>
            }
          />
          <StatCardIcon
            label="Benign"
            value={historyStats.benign}
            trend={historyStats.benignTrend}
            trendUp={false}
            iconBg="var(--color-accent-dim)"
            iconColor="var(--color-accent)"
            icon={
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 6 9 17l-5-5" />
              </svg>
            }
          />
          <StatCardIcon
            label="Avg Confidence"
            value={`${historyStats.avgConfidence}%`}
            trend={historyStats.avgConfidenceTrend}
            trendUp
            iconBg="var(--color-purple-dim)"
            iconColor="var(--color-purple)"
            icon={
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 12h4l2 8 4-16 2 8h6" />
              </svg>
            }
          />
        </div>

        <UploadPanel />

        <div className="flex flex-col gap-4 lg:flex-row">
          <ThreatTrendChart />
          <ThreatDistribution total={historyStats.maliciousDetected} />
        </div>

        <div className="flex flex-col gap-4 lg:flex-row">
          <TopTargetedFilesCard />
          <RiskBreakdownCard />
        </div>

        <RecentDetections />
      </main>
    </div>
  );
}
