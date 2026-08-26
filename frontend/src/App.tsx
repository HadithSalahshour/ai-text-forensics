import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api";
import { DiffPanel, Heatmap, ScoreHead, SignalList } from "./components";
import type { Analysis, Estimate, HumanizeResult, Status, Strength } from "./types";

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [mode, setMode] = useState<"paste" | "upload">("paste");
  const [text, setText] = useState("");
  const [dragging, setDragging] = useState(false);

  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [result, setResult] = useState<HumanizeResult | null>(null);
  const [busy, setBusy] = useState<"" | "analyze" | "humanize">("");
  const [error, setError] = useState("");

  const [strength, setStrength] = useState<Strength>("moderate");
  const [register, setRegister] = useState("");
  const [useLlm, setUseLlm] = useState(false);
  const [highQuality, setHighQuality] = useState(false);
  const [budget, setBudget] = useState(0.15);
  const [estimate, setEstimate] = useState<Estimate | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getStatus().then(setStatus).catch((e) => setError(String(e)));
  }, []);

  // Refresh the cost quote whenever anything that affects it changes, so the
  // number on screen always matches the run that's about to happen.
  useEffect(() => {
    if (!useLlm || !analysis) {
      setEstimate(null);
      return;
    }
    api
      .estimate(analysis.text, strength, highQuality)
      .then(setEstimate)
      .catch(() => setEstimate(null));
  }, [useLlm, highQuality, strength, analysis]);

  const runAnalyze = useCallback(async (payload?: File) => {
    setBusy("analyze");
    setError("");
    setResult(null);
    try {
      const data = payload ? await api.upload(payload) : await api.analyze(text);
      setAnalysis(data);
      setText(data.text);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy("");
    }
  }, [text]);

  const runHumanize = useCallback(async () => {
    if (!analysis) return;
    setBusy("humanize");
    setError("");
    try {
      setResult(
        await api.humanize({
          text: analysis.text,
          strength,
          register: register || null,
          useLlm,
          highQuality,
          budgetUsd: budget,
        }),
      );
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy("");
    }
  }, [analysis, strength, register, useLlm, highQuality, budget]);

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void runAnalyze(file);
  };

  const claudeOn = status?.privacy_mode === "claude";

  return (
    <div className="app">
      <header>
        <div>
          <h1>Humanized Writing</h1>
          <div className="sub">
            Local AI-writing detector and rewriter
          </div>
        </div>
        {status && (
          <div
            className={`badge ${claudeOn ? "claude" : "local"}`}
            title={
              claudeOn
                ? "The Claude rewrite pass is enabled. Detection is still fully local; only rewriting sends text to the API, and only when you tick the box."
                : "Nothing leaves this machine."
            }
          >
            <span className="dot" />
            {claudeOn ? "Claude API available" : "Local only — no network"}
          </div>
        )}
      </header>

      {error && <div className="error" style={{ marginBottom: 16 }}>{error}</div>}

      {/* ---------------- input ---------------- */}
      <section className="panel">
        <h2>Input</h2>
        <div className="tabs">
          <button
            className={`tab ${mode === "paste" ? "active" : ""}`}
            onClick={() => setMode("paste")}
          >
            Paste text
          </button>
          <button
            className={`tab ${mode === "upload" ? "active" : ""}`}
            onClick={() => setMode("upload")}
          >
            Upload file
          </button>
        </div>

        {mode === "paste" ? (
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste the writing you want to check…"
          />
        ) : (
          <div
            className={`dropzone ${dragging ? "over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInput.current?.click()}
          >
            <div style={{ fontSize: 15, marginBottom: 6 }}>
              Drop a file here, or click to choose
            </div>
            <div style={{ fontSize: 12.5 }}>
              {status?.supported_formats.join("  ·  ")}
            </div>
            <input
              ref={fileInput}
              type="file"
              hidden
              accept={status?.supported_formats.join(",")}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void runAnalyze(file);
              }}
            />
          </div>
        )}

        <div className="row" style={{ marginTop: 12 }}>
          <button
            className="primary"
            disabled={busy !== "" || (mode === "paste" && text.trim().length === 0)}
            onClick={() => void runAnalyze()}
          >
            {busy === "analyze" ? <><span className="spinner" /> Analyzing…</> : "Analyze"}
          </button>
          <button
            className="ghost"
            disabled={!text.trim()}
            onClick={() => void navigator.clipboard.writeText(text)}
          >
            Copy
          </button>
          <button
            className="ghost"
            disabled={!text.trim() && !analysis}
            onClick={() => {
              setText("");
              setAnalysis(null);
              setResult(null);
              setError("");
            }}
          >
            Clear
          </button>
          {text.trim() && (
            <span className="cost">{text.trim().split(/\s+/).length} words</span>
          )}
        </div>
      </section>

      {/* ---------------- results ---------------- */}
      {analysis && (
        <>
          <section className="panel">
            <h2>Result</h2>
            <ScoreHead analysis={analysis} />
          </section>

          <section className="panel">
            <h2>Where the score comes from</h2>
            <Heatmap text={analysis.text} sentences={analysis.sentences} />
          </section>

          <section className="panel">
            <h2>Strongest signals</h2>
            <SignalList signals={analysis.signals} />
          </section>

          {/* ---------------- humanize ---------------- */}
          <section className="panel">
            <h2>Humanize</h2>
            <div className="controls">
              <div className="field">
                <label>Strength</label>
                <select
                  value={strength}
                  onChange={(e) => setStrength(e.target.value as Strength)}
                >
                  <option value="light">Light — fewest changes</option>
                  <option value="moderate">Moderate — balanced</option>
                  <option value="aggressive">Aggressive — most changes</option>
                </select>
              </div>

              <div className="field">
                <label>Register</label>
                <select value={register} onChange={(e) => setRegister(e.target.value)}>
                  <option value="">Detect automatically</option>
                  <option value="academic">Academic</option>
                  <option value="professional">Professional</option>
                  <option value="casual">Casual</option>
                </select>
              </div>

              <div className="field">
                <label>Engine</label>
                <label
                  className="check"
                  title={status?.llm_available ? "" : status?.llm_reason}
                >
                  <input
                    type="checkbox"
                    checked={useLlm}
                    disabled={!status?.llm_available}
                    onChange={(e) => setUseLlm(e.target.checked)}
                  />
                  Also use Claude{" "}
                  <span className="cost">— recommended; local alone gets ~a third of the way</span>
                </label>
                {useLlm && (
                  <>
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={highQuality}
                        onChange={(e) => setHighQuality(e.target.checked)}
                      />
                      High quality ({status?.quality_model})
                    </label>
                    <label className="check">
                      Budget $
                      <input
                        type="number"
                        step="0.05"
                        min="0.01"
                        value={budget}
                        style={{ width: 74 }}
                        onChange={(e) => setBudget(Number(e.target.value))}
                      />
                    </label>
                  </>
                )}
              </div>

              {useLlm && (
                <div className="notice">
                  This sends the document to the Anthropic API — the only part of
                  this tool that uses the network.
                  {estimate &&
                    ` Up to ${estimate.max_calls} calls, roughly $${estimate.estimate_usd.toFixed(3)}, hard-capped at $${budget.toFixed(2)}.`}
                </div>
              )}
              {!status?.llm_available && (
                <div className="cost">
                  Local engine only. {status?.llm_reason}
                </div>
              )}

              <div className="row">
                <button
                  className="primary"
                  disabled={busy !== ""}
                  onClick={() => void runHumanize()}
                >
                  {busy === "humanize" ? (
                    <><span className="spinner" /> Rewriting…</>
                  ) : (
                    "Humanize"
                  )}
                </button>
                <span className="cost">
                  Target: below {status?.target_score ?? 20}
                </span>
              </div>
            </div>
          </section>
        </>
      )}

      {/* ---------------- humanize result ---------------- */}
      {result && (
        <section className="panel">
          <h2>Rewritten</h2>
          <DiffPanel result={result} />

          <div className="row" style={{ marginTop: 16 }}>
            <button
              className="ghost"
              onClick={() => void navigator.clipboard.writeText(result.final_text)}
            >
              Copy text
            </button>
            {(status?.export_formats ?? ["txt", "md", "docx", "pdf"]).map((format) => (
              <button
                key={format}
                className="ghost"
                onClick={() =>
                  void api
                    .exportText(result.final_text, format, analysis?.source_name)
                    .catch((e) => setError(String(e)))
                }
              >
                .{format}
              </button>
            ))}
            <button
              className="ghost"
              onClick={() =>
                void api.downloadScorecard(result.final_text).catch((e) => setError(String(e)))
              }
            >
              Score report
            </button>
            <button
              className="ghost"
              onClick={() => {
                setText(result.final_text);
                setAnalysis(null);
                setResult(null);
              }}
            >
              Re-analyze this version
            </button>
          </div>
        </section>
      )}

      <footer>
        Detection runs entirely on this machine. Scores are indicative — below{" "}
        {status?.min_reliable_words ?? 150} words the underlying statistics are too
        noisy for a confident verdict.
        <br />
        No detector, including this one, should be treated as proof of authorship.
      </footer>
    </div>
  );
}
