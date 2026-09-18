import type {
  AnalysisResult,
  HistoryEntry,
  RiskBreakdownSlice,
  ThreatDistributionSlice,
  TopTargetedFile,
} from "../types";

export const sampleFiles = ["memtest.exe", "setup.exe", "DW20.EXE"];

export const mockAnalysisResults: Record<string, AnalysisResult> = {
  "memtest.exe": {
    fileName: "memtest.exe",
    verdict: "MALICIOUS",
    risk: "CRITICAL",
    confidence: 94.7,
    md5: "631ea355665f28d4707448e442fbf5b8",
    features: {
      sections: 8,
      dllImports: 15,
      totalImports: 360,
      resources: 4,
      maxSectionEntropy: 7.22,
    },
    topFeatures: [
      { name: "SectionsMeanEntropy", impact: 0.31 },
      { name: "ImportsNb", impact: 0.24 },
      { name: "SectionsNb", impact: 0.18 },
      { name: "ResourcesMeanEntropy", impact: 0.12 },
      { name: "SizeOfCode", impact: 0.08 },
    ],
  },
  "setup.exe": {
    fileName: "setup.exe",
    verdict: "BENIGN",
    risk: "LOW",
    confidence: 98.1,
    md5: "a13c9f0e2b8d4462a9e5d8b6f1c0d7a4",
    features: {
      sections: 6,
      dllImports: 9,
      totalImports: 142,
      resources: 12,
      maxSectionEntropy: 5.41,
    },
    topFeatures: [
      { name: "ImportsNb", impact: 0.19 },
      { name: "ResourcesNb", impact: 0.14 },
      { name: "SectionsMeanEntropy", impact: 0.11 },
      { name: "SizeOfCode", impact: 0.07 },
      { name: "SectionsNb", impact: 0.05 },
    ],
  },
  "DW20.EXE": {
    fileName: "DW20.EXE",
    verdict: "MALICIOUS",
    risk: "HIGH",
    confidence: 88.3,
    md5: "f4e2a9c1d6b3487fae90c2b1d5e8a7f3",
    features: {
      sections: 7,
      dllImports: 13,
      totalImports: 281,
      resources: 3,
      maxSectionEntropy: 6.84,
    },
    topFeatures: [
      { name: "SectionsMeanEntropy", impact: 0.27 },
      { name: "ImportsNb", impact: 0.21 },
      { name: "ResourcesMeanEntropy", impact: 0.16 },
      { name: "SectionsNb", impact: 0.1 },
      { name: "SizeOfCode", impact: 0.06 },
    ],
  },
};

export const historyStats = {
  totalAnalyses: 1284,
  totalAnalysesTrend: "+12.4%",
  maliciousDetected: 437,
  maliciousTrend: "+8.1%",
  benign: 847,
  benignTrend: "-3.2%",
  avgConfidence: 91.4,
  avgConfidenceTrend: "+2.6%",
};

export const historyEntries: HistoryEntry[] = [
  { fileName: "memtest.exe", verdict: "MALICIOUS", confidence: 94.7, risk: "CRITICAL", time: "10:42", action: "Isolated in sandbox" },
  { fileName: "setup.exe", verdict: "BENIGN", confidence: 98.1, risk: "LOW", time: "10:45", action: "Cleared for release" },
  { fileName: "DW20.EXE", verdict: "MALICIOUS", confidence: 88.3, risk: "HIGH", time: "10:49", action: "Queued for deep scan" },
  { fileName: "ose.exe", verdict: "BENIGN", confidence: 96.5, risk: "LOW", time: "10:52", action: "Archived to baseline" },
  { fileName: "installer_x64.exe", verdict: "MALICIOUS", confidence: 76.2, risk: "MEDIUM", time: "10:58", action: "Analyst review requested" },
];

export const threatDistribution: ThreatDistributionSlice[] = [
  { label: "Malware", pct: 32, color: "var(--color-warning)" },
  { label: "Phishing", pct: 23, color: "var(--color-info)" },
  { label: "Ransomware", pct: 20, color: "var(--color-danger)" },
  { label: "Trojan", pct: 16, color: "var(--color-purple)" },
  { label: "Spyware", pct: 9, color: "var(--color-accent-light)" },
];

export const riskBreakdown: RiskBreakdownSlice[] = [
  { level: "CRITICAL", count: 128, color: "var(--color-danger)" },
  { level: "HIGH", count: 184, color: "var(--color-warning)" },
  { level: "MEDIUM", count: 211, color: "var(--color-gold)" },
  { level: "LOW", count: 761, color: "var(--color-accent-light)" },
];

export const responseStats = {
  avgResponse: "4m 12s",
  escalations: 19,
};

export const topTargetedFiles: TopTargetedFile[] = [
  { fileName: "memtest.exe", confidence: 94.7, color: "var(--color-warning)" },
  { fileName: "installer_x64.exe", confidence: 76.2, color: "var(--color-danger)" },
  { fileName: "DW20.EXE", confidence: 88.3, color: "var(--color-gold)" },
  { fileName: "setup.exe", confidence: 98.1, color: "var(--color-accent-light)" },
];
