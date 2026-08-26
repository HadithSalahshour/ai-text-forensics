import type { Analysis, Estimate, HumanizeResult, Status, Strength } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* response wasn't JSON; keep the status text */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getStatus = () => request<Status>("/api/status");

export const analyze = (text: string) =>
  request<Analysis>("/api/analyze", json({ text }));

export const upload = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return request<Analysis>("/api/upload", { method: "POST", body: form });
};

export const estimate = (text: string, strength: Strength, highQuality: boolean) =>
  request<Estimate>(
    "/api/estimate",
    json({ text, strength, high_quality: highQuality }),
  );

export const humanize = (options: {
  text: string;
  strength: Strength;
  register: string | null;
  useLlm: boolean;
  highQuality: boolean;
  budgetUsd: number;
}) =>
  request<HumanizeResult>(
    "/api/humanize",
    json({
      text: options.text,
      strength: options.strength,
      register: options.register,
      use_llm: options.useLlm,
      high_quality: options.highQuality,
      budget_usd: options.budgetUsd,
    }),
  );

/** Download a blob response without leaving the page. */
async function download(path: string, body: unknown, filename: string) {
  const response = await fetch(path, json(body));
  if (!response.ok) throw new Error(await response.text());
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export const exportText = (text: string, format: string, title?: string) =>
  download("/api/export", { text, format, title }, `humanized.${format}`);

export const downloadScorecard = (text: string) =>
  download("/api/scorecard", { text }, "report.md");
