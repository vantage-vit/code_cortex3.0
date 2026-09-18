export type Verdict = "MALICIOUS" | "BENIGN";

export type RiskLevel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

export interface PeFeatures {
  sections: number;
  dllImports: number;
  totalImports: number;
  resources: number;
  maxSectionEntropy: number;
}

export interface FeatureContribution {
  name: string;
  impact: number;
}

export interface AnalysisResult {
  fileName: string;
  verdict: Verdict;
  risk: RiskLevel;
  confidence: number;
  md5: string;
  features: PeFeatures;
  topFeatures: FeatureContribution[];
}

export interface HistoryEntry {
  fileName: string;
  verdict: Verdict;
  confidence: number;
  risk: RiskLevel;
  time: string;
  action: string;
}

export interface ThreatDistributionSlice {
  label: string;
  pct: number;
  color: string;
}

export interface RiskBreakdownSlice {
  level: RiskLevel;
  count: number;
  color: string;
}

export interface TopTargetedFile {
  fileName: string;
  confidence: number;
  color: string;
}
