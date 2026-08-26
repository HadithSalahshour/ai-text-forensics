export interface Signal {
  key: string;
  label: string;
  tier: string;
  value: number;
  score: number;
  weight: number;
  evidence: string;
}

export interface SentenceScore {
  index: number;
  text: string;
  start: number;
  end: number;
  score: number;
  reasons: string[];
}

export interface Analysis {
  text: string;
  source_name: string;
  score: number;
  verdict: string;
  reliable: boolean;
  note: string;
  word_count: number;
  sentence_count: number;
  paragraph_count: number;
  tier_scores: Record<string, number>;
  signals: Signal[];
  sentences: SentenceScore[];
  statistical_available: boolean;
  fitted_used: boolean;
}

export interface Edit {
  original: string;
  rewritten: string;
  transform: string;
  note: string;
  similarity: number | null;
}

export interface HumanizeResult {
  original_text: string;
  final_text: string;
  score_before: number;
  score_after: number;
  verdict_before: string;
  verdict_after: string;
  target_met: boolean;
  document_similarity: number | null;
  edits: Edit[];
  notes: string[];
  tier_before: Record<string, number>;
  tier_after: Record<string, number>;
  llm_used: boolean;
  llm_cost_usd: number;
  llm_usage_summary: string;
}

export interface Status {
  privacy_mode: "local" | "claude";
  llm_available: boolean;
  llm_reason: string;
  llm_model: string;
  quality_model: string;
  statistical_available: boolean;
  statistical_error: string | null;
  spacy_available: boolean;
  calibrated_model: string | null;
  calibrated_in_use: boolean;
  target_score: number;
  min_reliable_words: number;
  supported_formats: string[];
  export_formats: string[];
}

export interface Estimate {
  max_calls: number;
  estimate_usd: number;
  model: string;
  available: boolean;
}

export type Strength = "light" | "moderate" | "aggressive";
