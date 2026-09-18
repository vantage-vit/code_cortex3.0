import type { AnalysisResult, HistoryEntry } from "../types";

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
  maliciousDetected: 437,
  benign: 847,
};

export const historyEntries: HistoryEntry[] = [
  { fileName: "memtest.exe", verdict: "MALICIOUS", confidence: 94.7, risk: "CRITICAL", time: "10:42" },
  { fileName: "setup.exe", verdict: "BENIGN", confidence: 98.1, risk: "LOW", time: "10:45" },
  { fileName: "DW20.EXE", verdict: "MALICIOUS", confidence: 88.3, risk: "HIGH", time: "10:49" },
  { fileName: "ose.exe", verdict: "BENIGN", confidence: 96.5, risk: "LOW", time: "10:52" },
  { fileName: "installer_x64.exe", verdict: "MALICIOUS", confidence: 76.2, risk: "MEDIUM", time: "10:58" },
];
