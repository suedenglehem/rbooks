/**
 * Typed client for the research API (src/library_rag/api.py + reader.py).
 *
 * All requests are same-origin: the server also serves this UI's own bundle.
 * When the service has a bearer token configured, it is attached to every
 * request — including the /books/.../source fetch the PDF reader needs,
 * because pdf.js's internal Range requests cannot carry headers. The bytes
 * are therefore fetched here (with auth) and handed to pdf.js as data.
 */

// --- shapes ----------------------------------------------------------------

export interface Book {
  doc_id: string;
  title: string;
  format: string;
  rev_id: string;
  size_bytes: number;
  published: boolean;
  chunk_count: number;
}

export interface PassageSpan {
  unit_id: string;
  source_start: number;
  source_end: number;
  bbox: number[] | null;
}

export interface Passage {
  chunk_id: string;
  doc_id: string;
  rev_id: string;
  title: string | null;
  text: string;
  score: number;
  dense_rank: number | null;
  sparse_rank: number | null;
  spans: PassageSpan[];
}

export interface SearchResponse {
  query: string;
  degraded: boolean;
  degraded_reason: string | null;
  counts: Record<string, number>;
  passages: Passage[];
}

export interface EvidenceBox {
  unit_id: string;
  page: number | null; // 1-based physical page, or null for sections
  bbox: number[] | null; // page points, bottom-left origin (unrotated)
}

export interface EvidenceLocation {
  kind: "page" | "section" | "unknown";
  page?: number | null; // 1-based
  label?: string | null;
  pages?: { page: number; label: string | null }[] | null;
  ref?: string | null;
  anchor?: string | null;
  title?: string | null;
}

export interface Evidence {
  evidence_id: string;
  chunk_id: string;
  doc_id: string;
  rev_id: string;
  run_id: string;
  text: string;
  title: string | null;
  source_title: string;
  format: string;
  location: EvidenceLocation;
  boxes: EvidenceBox[];
  quality_flags: string[];
}

export interface AnswerRow {
  answer_id: string;
  created_at: number;
  query: string;
  doc_id: string | null;
  rev_id: string | null;
  status: "answered" | "abstained" | "failed";
  model_revision: string | null;
  prompt_version: string | null;
  evidence: Evidence[];
  answer_text: string | null;
  abstain_reason: string | null;
  failure_reason: string | null;
  citations: string[];
  search_counts: Record<string, number> | null;
  retrieval_ms: number | null;
  model_ms: number | null;
}

export interface AnswerSummary {
  answer_id: string;
  created_at: number;
  query: string;
  status: string;
  citation_count: number;
}

export interface CitationResolution {
  available: boolean;
  reason: string | null;
  excerpt: string | null;
  reader: { manifest: string; source: string } | null;
}

export interface UnitRef {
  unit_id: string;
  position: number; // zero-based
  kind: "page" | "section";
  ref: string | null;
  char_count: number;
  quality_flags: string[];
  route: string | null;
  ocr_state: string | null;
}

export interface BookManifest {
  rev_id: string;
  doc_id: string;
  format: string;
  run_id: string;
  unit_count: number;
  units: UnitRef[];
  // Present only when the server is configured with
  // services.show_path_to_original: true (local-machine opt-in).
  source_path?: string | null;
}

export interface UnitPayload {
  kind: "page" | "section";
  position: number;
  label?: string | null;
  ref?: string | null;
  // page units:
  rotation?: number;
  width?: number;
  height?: number;
  text?: string;
  quality?: { chars: number; flags: string[] };
  pdfjs_page?: { page: number; label: string | null };
  // section units:
  title?: string | null;
  paragraphs?: { anchor: string; text: string }[];
  sanitized_html?: string;
  char_count?: number;
}

export interface IngestStatus {
  paused: boolean;
  jobs: Record<string, number | string>;
  schema_version: number;
  documents: number;
  revisions: number;
  chunks: number;
  answers: number;
  resumes: number;
}

export interface ScanReport {
  root: string;
  mount_unavailable: boolean;
  discovered: number;
  unchanged: number;
  new_documents: number;
  new_revisions: number;
  aliases: unknown[];
  invalid: unknown[];
  missing: unknown[];
  changed_during_scan: unknown[];
  jobs_enqueued: number;
}

export interface ResumeSummary {
  rev_id: string;
  doc_id: string;
  title: string;
  score: number; // bm25: lower is better
  excerpt: string;
}

export interface ResumeRecord {
  rev_id: string;
  doc_id: string;
  title: string | null;
  text: string;
  word_count: number;
  model_revision: string;
  prompt_version: string;
  updated_at: number;
}

// M10: one level of the configured books root. `path` is relative to the
// root; the server never accepts absolute paths or escapes. Files carry the
// catalog's active rev when indexed (null fields = "not in the index yet").
export interface BrowseEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size_bytes: number | null;
  rev_id: string | null;
  title: string | null;
}

// --- token ------------------------------------------------------------------

let token = readStoredToken();

function readStoredToken(): string {
  try {
    return sessionStorage.getItem("library_rag_token") ?? "";
  } catch {
    return "";
  }
}

export function getToken(): string {
  return token;
}

export function setToken(t: string): void {
  token = t.trim();
  try {
    if (token) sessionStorage.setItem("library_rag_token", token);
    else sessionStorage.removeItem("library_rag_token");
  } catch {
    // private mode: keep the in-memory token only
  }
}

// --- transport ----------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function errText(e: unknown): string {
  if (e instanceof ApiError) return `HTTP ${e.status}: ${e.message}`;
  return e instanceof Error ? e.message : String(e);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    ...((init.headers as Record<string, string> | undefined) ?? {}),
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(path, { ...init, headers });
  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!res.ok) {
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText || `HTTP ${res.status}`;
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

function post(path: string, data: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
}

// --- endpoints ----------------------------------------------------------------

export const api = {
  health: () => request<{ status: string; version: string }>("/health"),
  ready: () =>
    request<{
      qdrant: boolean;
      embedding_model: boolean;
      answer_model: boolean;
      token_required: boolean;
      browse: boolean;
    }>("/ready"),
  library: () => request<{ books: Book[] }>("/library"),

  search: (q: {
    query: string;
    doc?: string;
    rev?: string;
    limit?: number;
  }) => request<SearchResponse>("/search", post("/search", q)),

  answer: (q: { query: string; doc?: string; rev?: string }) =>
    request<AnswerRow>("/answer", post("/answer", q)),

  answers: (limit = 20) =>
    request<{ answers: AnswerSummary[] }>(`/answers?limit=${limit}`),
  oneAnswer: (id: string) =>
    request<AnswerRow>(`/answers/${encodeURIComponent(id)}`),
  citation: (answerId: string, evidenceId: string) =>
    request<CitationResolution>(
      `/answers/${encodeURIComponent(answerId)}/citations/${encodeURIComponent(evidenceId)}`,
    ),

  book: (revId: string) =>
    request<BookManifest>(`/books/${encodeURIComponent(revId)}`),
  unit: (revId: string, unitId: string) =>
    request<UnitPayload>(
      `/books/${encodeURIComponent(revId)}/units/${encodeURIComponent(unitId)}`,
    ),
  sourceUrl: (revId: string) => `/books/${encodeURIComponent(revId)}/source`,
  async sourceBytes(revId: string): Promise<ArrayBuffer> {
    const res = await fetch(api.sourceUrl(revId), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) {
      throw new ApiError(res.status, `source unavailable (HTTP ${res.status})`);
    }
    return await res.arrayBuffer();
  },

  searchResumes: (query: string, limit = 20) =>
    request<{ results: ResumeSummary[] }>("/resumes/search", post("/resumes/search", { query, limit })),
  resume: (revId: string) =>
    request<ResumeRecord>(`/resumes/${encodeURIComponent(revId)}`),
  browseDir: (path: string) =>
    request<{ path: string; entries: BrowseEntry[] }>(
      `/browse/dir?path=${encodeURIComponent(path)}`,
    ),
  ingestStatus: () => request<IngestStatus>("/ingest/status"),
  scan: () => request<{ reports: ScanReport[] }>("/scan", post("/scan", {})),
  ingestPause: (reason: string) =>
    request<{ paused: boolean }>("/ingest/pause", post("/ingest/pause", { reason })),
  ingestResume: () =>
    request<{ paused: boolean }>("/ingest/resume", post("/ingest/resume", {})),
  ingestRetry: (includePermanent: boolean) =>
    request<{ requeued: number }>("/ingest/retry", post("/ingest/retry", { include_permanent: includePermanent })),
};

/** Authed download of the archived original (blob URL + temporary anchor). */
export async function downloadSource(revId: string, filename: string): Promise<void> {
  const res = await fetch(api.sourceUrl(revId), {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new ApiError(res.status, `download failed (HTTP ${res.status})`);
  const blob = new Blob([await res.arrayBuffer()], {
    type: res.headers.get("Content-Type") ?? "application/octet-stream",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
