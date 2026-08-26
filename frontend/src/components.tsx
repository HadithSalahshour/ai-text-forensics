import { useState } from "react";
import type { Analysis, HumanizeResult, SentenceScore, Signal } from "./types";

/** Score → colour. Shared by the gauge, tier bars, and heatmap so a given
 *  number always looks the same wherever it appears. */
export function bandColor(score: number): string {
  if (score < 20) return "var(--good)";
  if (score < 40) return "var(--ok)";
  if (score < 60) return "var(--warn)";
  if (score < 80) return "var(--bad)";
  return "var(--worst)";
}

export function TierBars({ tiers }: { tiers: Record<string, number> }) {
  const order = ["statistical", "structure", "lexicon", "mechanics", "syntax"];
  const labels: Record<string, string> = {
    statistical: "Statistical",
    structure: "Structure",
    lexicon: "Lexicon",
    mechanics: "Mechanics",
    syntax: "Syntax",
  };
  return (
    <div className="tiers">
      {order
        .filter((key) => key in tiers)
        .map((key) => (
          <div className="tier" key={key}>
            <div className="name">{labels[key]}</div>
            <div className="track">
              <div
                className="fill"
                style={{ width: `${tiers[key]}%`, background: bandColor(tiers[key]) }}
              />
            </div>
            <div className="val">{tiers[key].toFixed(0)}</div>
          </div>
        ))}
    </div>
  );
}

export function ScoreHead({ analysis }: { analysis: Analysis }) {
  return (
    <>
      <div className="score-head">
        <div>
          <div className="score-number" style={{ color: bandColor(analysis.score) }}>
            {analysis.score.toFixed(1)}
            <span>/100</span>
          </div>
          <div className="verdict" style={{ color: bandColor(analysis.score) }}>
            {analysis.verdict}
          </div>
          <div className="meta">
            {analysis.word_count} words · {analysis.sentence_count} sentences ·{" "}
            {analysis.paragraph_count} paragraphs
          </div>
        </div>
        <TierBars tiers={analysis.tier_scores} />
      </div>

      {analysis.note && <div className="notice">{analysis.note}</div>}
      {!analysis.statistical_available && (
        <div className="notice">
          The statistical tier is unavailable, so this score rests on writing style
          alone. It will under-detect machine text that has been lightly edited.
        </div>
      )}
    </>
  );
}

/** Renders the document with each sentence tinted by its score.
 *
 *  Sentences are painted using their character offsets rather than by
 *  re-splitting the text in the browser, so what you see highlighted is exactly
 *  the span the backend scored. Gaps between sentences (whitespace, paragraph
 *  breaks) are emitted verbatim to preserve the document's shape. */
export function Heatmap({
  text,
  sentences,
}: {
  text: string;
  sentences: SentenceScore[];
}) {
  const [selected, setSelected] = useState<SentenceScore | null>(null);

  const nodes: React.ReactNode[] = [];
  let cursor = 0;

  sentences.forEach((sentence) => {
    if (sentence.start > cursor) {
      nodes.push(<span key={`gap-${sentence.index}`}>{text.slice(cursor, sentence.start)}</span>);
    }
    const intensity = Math.min(sentence.score / 100, 1);
    nodes.push(
      <span
        key={sentence.index}
        className="sent"
        title={sentence.reasons.join(" · ") || "No specific markers"}
        onClick={() => setSelected(sentence)}
        style={{
          background:
            selected?.index === sentence.index
              ? "color-mix(in srgb, var(--accent) 26%, transparent)"
              : intensity > 0.12
                ? `color-mix(in srgb, ${bandColor(sentence.score)} ${Math.round(intensity * 34)}%, transparent)`
                : "transparent",
          borderBottomColor:
            selected?.index === sentence.index
              ? "var(--accent)"
              : intensity > 0.4
                ? bandColor(sentence.score)
                : "transparent",
        }}
      >
        {text.slice(sentence.start, sentence.end)}
      </span>,
    );
    cursor = sentence.end;
  });

  if (cursor < text.length) nodes.push(<span key="tail">{text.slice(cursor)}</span>);

  return (
    <>
      <div className="legend">
        <span>Click a sentence for detail</span>
        {[
          ["Human-like", 10],
          ["Mild", 30],
          ["Moderate", 50],
          ["Strong", 70],
          ["Very strong", 90],
        ].map(([label, value]) => (
          <span key={label as string}>
            <span
              className="swatch"
              style={{ background: bandColor(value as number) }}
            />
            {label}
          </span>
        ))}
      </div>

      {/* Above the text, not below it. On a long document the panel was
          rendering hundreds of pixels off-screen, so clicking a sentence
          looked like it did nothing. */}
      {selected && (
        <div className="reasons" style={{ marginTop: 0, marginBottom: 14 }}>
          <strong>
            Sentence {selected.index + 1} — {selected.score.toFixed(0)}/100
          </strong>
          {selected.reasons.length ? (
            <ul>
              {selected.reasons.map((reason, i) => (
                <li key={i}>{reason}</li>
              ))}
            </ul>
          ) : (
            <span>No specific machine-writing markers in this sentence.</span>
          )}
        </div>
      )}

      <div className="heatmap" style={{ whiteSpace: "pre-wrap" }}>
        {nodes}
      </div>
    </>
  );
}

export function SignalList({ signals }: { signals: Signal[] }) {
  return (
    <div>
      {signals.map((signal) => (
        <div className="signal" key={signal.key}>
          <div>
            <div>{signal.label}</div>
            <div className="evidence">{signal.evidence}</div>
          </div>
          <div className="track" style={{ width: 90 }}>
            <div
              className="fill"
              style={{
                width: `${signal.score * 100}%`,
                background: bandColor(signal.score * 100),
              }}
            />
          </div>
          <div className="val" style={{ textAlign: "right" }}>
            {(signal.score * 100).toFixed(0)}%
          </div>
        </div>
      ))}
    </div>
  );
}

export function DiffPanel({ result }: { result: HumanizeResult }) {
  const [showEdits, setShowEdits] = useState(false);

  return (
    <>
      <div className="arrow">
        <span style={{ color: bandColor(result.score_before) }}>
          {result.score_before.toFixed(1)}
        </span>
        <span className="sep">→</span>
        <span style={{ color: bandColor(result.score_after) }}>
          {result.score_after.toFixed(1)}
        </span>
        <span className="cost" style={{ fontSize: 13 }}>
          {result.target_met ? "target met" : "above target"} ·{" "}
          {result.edits.length} edits
          {result.document_similarity !== null &&
            ` · meaning preserved ${result.document_similarity.toFixed(3)}`}
          {result.llm_cost_usd > 0 && ` · $${result.llm_cost_usd.toFixed(4)}`}
        </span>
      </div>

      {/* Local-only output is a partial result, and saying so is the whole
          point of this banner. Measured against a held-out judge the local
          transforms close roughly a third of the gap to human statistics,
          while the Claude pass closes essentially all of it. Presenting the
          local number without that context is how someone ends up pasting a
          half-finished rewrite into a real detector and being surprised. */}
      {!result.llm_used && (
        <div className="notice" style={{ marginTop: 12 }}>
          <strong>This is a partial rewrite.</strong> The local engine changes
          wording and sentence rhythm, but it cannot move the statistical
          signature far — on a held-out test it closed about a third of the gap
          to human-level text, while the Claude pass closed essentially all of
          it. Real detectors will very likely still flag this. Tick{" "}
          <em>Also use Claude</em> and re-run for a result worth testing.
        </div>
      )}

      {result.notes.map((note, i) => (
        <div
          key={i}
          className={note.includes("WARNING") || note.includes("revert") ? "notice" : "meta"}
          style={{ marginTop: 8 }}
        >
          {note}
        </div>
      ))}

      <div className="diff" style={{ marginTop: 16 }}>
        <div className="diff-col">
          <h3>Original</h3>
          <div className="diff-body">{result.original_text}</div>
        </div>
        <div className="diff-col">
          <h3>Humanized</h3>
          <div className="diff-body">{result.final_text}</div>
        </div>
      </div>

      {result.edits.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <button className="ghost" onClick={() => setShowEdits((v) => !v)}>
            {showEdits ? "Hide" : "Show"} {result.edits.length} edits
          </button>
          {showEdits && (
            <div style={{ marginTop: 12 }}>
              {result.edits.map((edit, i) => (
                <div className="edit-item" key={i}>
                  <div className="tag">
                    {edit.transform}
                    {edit.similarity !== null && ` · sim ${edit.similarity.toFixed(3)}`}
                  </div>
                  <div className="del">{edit.original}</div>
                  <div className="ins">{edit.rewritten}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );
}
