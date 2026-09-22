/**
 * Application shell (PRD §10): a two-pane research interface — query/answer
 * on the left, source reader on the right — plus an ingestion dashboard and
 * a resumes view (keyword search over the stored per-book summaries, M8).
 * Below 980px the panes collapse into Query/Reader tabs. No framework: plain
 * TypeScript building the DOM, talking to the same-origin research API.
 */
import {
  api,
  ApiError,
  downloadSource,
  errText,
  getToken,
  setToken,
} from "./api";
import type {
  AnswerRow,
  Book,
  BookManifest,
  Evidence,
  EvidenceLocation,
  Passage,
  PassageSpan,
  ResumeSummary,
  SearchResponse,
} from "./api";
import { Reader } from "./reader";
import type { ReaderState } from "./reader";

// --- small DOM helpers ------------------------------------------------------

type Child = Node | string | null | undefined;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | null> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null) node.setAttribute(k, v);
  }
  for (const c of children) {
    if (c !== null && c !== undefined && c !== "") node.append(c);
  }
  return node;
}

function button(label: string, cls: string | null, fn: () => void | Promise<void>): HTMLButtonElement {
  const b = el("button", cls ? { class: cls } : {}, label);
  b.addEventListener("click", () => {
    void fn();
  });
  return b;
}

function locationLabel(loc: EvidenceLocation): string {
  if (loc.kind === "page") {
    if (loc.page === null || loc.page === undefined) return "page (unnumbered)";
    return loc.label ? `page ${loc.page} · ${loc.label}` : `page ${loc.page}`;
  }
  if (loc.kind === "section") return loc.title || loc.ref || "section";
  return "location unknown";
}

function timeAgo(ts: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

// --- app ---------------------------------------------------------------------

export class App {
  private reader: Reader;
  private books: Book[] = [];
  private bookFilter: { doc: string; rev: string } | null = null;
  private mode: "answer" | "search" = "answer";
  private running = false;
  private currentAnswer: AnswerRow | null = null;
  private manifests = new Map<string, Promise<BookManifest | null>>();
  private resolvedManifests = new Map<string, BookManifest>();
  private readyTimer = 0;
  private resizeT = 0;
  // null = unknown (before the first /ready); then the server's own word on
  // whether bearer auth is enforced. The token widget (field + "Set" button)
  // is only ever shown while a token may be needed.
  private tokenRequired: boolean | null = null;
  // The resume currently shown in the resumes view (rev + fallback title),
  // so "Open book" knows which revision to deep-link into the reader.
  private resumeState: { rev: string; title: string } | null = null;

  constructor(private root: HTMLElement) {
    root.innerHTML = "";
    root.append(this.buildHeader(), el("main", {}, this.buildResearchView(), this.buildIngestView(), this.buildResumeView()));

    const canvas = root.querySelector<HTMLCanvasElement>("#pdf-canvas");
    const epub = root.querySelector<HTMLElement>("#epub-host");
    const scroll = root.querySelector<HTMLElement>("#reader-scroll");
    if (canvas === null || epub === null || scroll === null) throw new Error("reader elements missing");
    this.reader = new Reader(canvas, epub, scroll, (s) => this.onReaderState(s));

    if (!window.matchMedia("(min-width: 980px)").matches) {
      root.querySelector<HTMLElement>("#view-research")?.classList.add("show-query");
    }
    window.addEventListener("resize", () => {
      window.clearTimeout(this.resizeT);
      this.resizeT = window.setTimeout(() => void this.reader.refresh(), 150);
    });
    const mq = window.matchMedia("(min-width: 980px)");
    mq.addEventListener("change", () => {
      if (mq.matches) {
        this.root.querySelector<HTMLElement>("#view-research")?.classList.remove("show-query", "show-reader");
      }
      void this.reader.refresh();
    });
    void this.startup();
  }

  // --- construction -----------------------------------------------------------

  private buildHeader(): HTMLElement {
    const readiness = el("div", { class: "readiness" },
      this.dot("qdrant", "Qdrant"),
      this.dot("embedding", "Embeddings"),
      this.dot("answer", "Answer model"),
    );
    const views = el("nav", { class: "views" },
      this.navBtn("research", "Research"),
      this.navBtn("ingest", "Ingestion"),
      this.navBtn("resumes", "Resumes"),
    );
    const tokenWrap = el("div", { class: "token", hidden: "true" },
      el("label", { class: "token-label" }, "API token",
        el("input", { type: "password", id: "token-input", autocomplete: "off" })),
      button("Set", "btn small", () => this.saveToken()),
    );
    return el("header", { class: "app-header" },
      el("h1", { class: "title" }, "Library Research"),
      readiness,
      views,
      tokenWrap,
    );
  }

  private dot(id: string, label: string): HTMLElement {
    return el("span", { class: "dot unknown", id: `ready-${id}` },
      el("span", { class: "dot-marker" }), label);
  }

  private navBtn(view: "research" | "ingest" | "resumes", label: string): HTMLButtonElement {
    const b = el("button", { class: "view-btn", "data-view": view }, label);
    b.addEventListener("click", () => this.showView(view));
    return b;
  }

  private buildResearchView(): HTMLElement {
    const tabs = el("div", { class: "narrow-tabs" },
      this.paneBtn("query", "Query"),
      this.paneBtn("reader", "Reader"),
    );

    const bookSel = el("select", { id: "book-filter" }, el("option", { value: "" }, "All books"));
    bookSel.addEventListener("change", () => this.onBookFilterChange(bookSel));
    const modeBtnA = el("button", { class: "mode-btn active", "data-mode": "answer" }, "Answer");
    const modeBtnS = el("button", { class: "mode-btn", "data-mode": "search" }, "Search only");
    modeBtnA.addEventListener("click", () => this.setMode("answer"));
    modeBtnS.addEventListener("click", () => this.setMode("search"));
    const query = el("textarea", { id: "query", rows: "3" },
      "Ask a question across your library…");
    query.value = "";
    query.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        void this.run();
      }
    });
    const runBtn = button(this.mode === "answer" ? "Ask" : "Search", "btn primary big", () => this.run());
    runBtn.id = "run-btn";

    const queryPane = el("div", { class: "pane pane-query" },
      el("div", { class: "controls" },
        bookSel,
        el("div", { class: "mode-toggle" }, modeBtnA, modeBtnS),
      ),
      query,
      runBtn,
      el("div", { id: "results" }),
      el("h2", { class: "section-h" }, "Answer history"),
      el("ul", { class: "history", id: "history" }),
    );

    const rPrev = button("←", "btn icon", () => this.readerStep(-1));
    rPrev.id = "r-prev";
    rPrev.setAttribute("aria-label", "Previous");
    const rNext = button("→", "btn icon", () => this.readerStep(1));
    rNext.id = "r-next";
    rNext.setAttribute("aria-label", "Next");
    const toolbar = el("div", { class: "reader-toolbar" },
      rPrev,
      el("span", { class: "reader-pos", id: "r-pos" }, "—"),
      rNext,
      el("span", { class: "spacer" }),
      button("−", "btn icon", () => this.reader.zoomBy(1 / 1.25)),
      button("Fit", "btn", () => this.reader.zoomFit()),
      button("+", "btn icon", () => this.reader.zoomBy(1.25)),
      button("Download original", "btn", () => this.downloadCurrent()),
    );

    const scroll = el("div", { class: "reader-scroll", id: "reader-scroll" },
      el("div", { class: "pdf-wrap" }, el("canvas", { id: "pdf-canvas" })),
      el("article", { class: "epub", id: "epub-host", hidden: "true" }),
      el("div", { class: "reader-unavailable", id: "reader-unavail", hidden: "true" }),
      el("div", { class: "reader-empty", id: "reader-empty" },
        "Open a book here — click any Locate button or citation."),
    );

    const readerPane = el("div", { class: "pane pane-reader" },
      toolbar,
      el("div", { class: "reader-path", id: "r-path", hidden: "true" }),
      scroll, el("div", { class: "reader-status", id: "reader-status" }));

    return el("section", { id: "view-research" },
      tabs,
      el("div", { class: "research-grid" }, queryPane, readerPane),
    );
  }

  private paneBtn(pane: string, label: string): HTMLButtonElement {
    const b = el("button", { class: "narrow-tab", "data-pane": pane }, label);
    b.addEventListener("click", () => this.setNarrowPane(pane as "query" | "reader"));
    return b;
  }

  private card(id: string, label: string): HTMLElement {
    return el("div", { class: "card" },
      el("div", { class: "card-k" }, label),
      el("div", { class: "card-v", id }, "…"));
  }

  private buildIngestView(): HTMLElement {
    const cards = el("div", { class: "cards" },
      this.card("i-docs", "Documents"),
      this.card("i-revs", "Revisions"),
      this.card("i-chunks", "Chunks"),
      this.card("i-answers", "Answers"),
      this.card("i-schema", "Schema"),
      this.card("i-paused", "Ingestion"),
    );
    const actions = el("div", { class: "ingest-actions" },
      button("Scan folders", "btn primary", () => this.ingestScan()),
      button("Pause", "btn", () => this.ingestPause()),
      button("Resume", "btn", () => this.ingestResume()),
      button("Retry failed", "btn", () => this.ingestRetry(false)),
      button("Retry all", "btn danger", () => this.ingestRetry(true)),
    );
    return el("section", { id: "view-ingest", hidden: "true" },
      el("div", { class: "ingest-grid" },
        cards,
        actions,
        el("div", { class: "ingest-msg", id: "i-msg" }),
        el("h3", { class: "section-h" }, "Jobs"),
        el("table", { class: "jobs", id: "i-jobs" }),
        el("h3", { class: "section-h" }, "Last scan"),
        el("pre", { class: "scan-report", id: "i-scan-report" }, "No scan yet in this session."),
      ));
  }

  private buildResumeView(): HTMLElement {
    const query = el("input", {
      type: "search",
      id: "resume-query",
      placeholder: "Keywords — e.g. lighthouse, thermodynamics, ledgers…",
    });
    query.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        void this.resumeSearch();
      }
    });
    const left = el("div", { class: "pane" },
      el("div", { class: "controls" }, query, button("Search", "btn primary", () => this.resumeSearch())),
      el("div", { class: "ingest-msg", id: "resume-msg" }),
      el("div", { id: "resume-results" }),
    );
    const right = el("div", { class: "pane" },
      el("div", { class: "controls" },
        el("h3", { class: "section-h", id: "resume-title" }, "No resume selected"),
        el("span", { class: "spacer" }),
        button("Open book", "btn", () => this.openResumeBook()),
      ),
      el("div", { class: "ingest-msg", id: "resume-meta" }),
      el("div", { class: "resume-text", id: "resume-text" }),
    );
    return el("section", { id: "view-resumes", hidden: "true" },
      el("div", { class: "research-grid" }, left, right));
  }

  // --- startup / views ---------------------------------------------------------

  private async startup(): Promise<void> {
    await Promise.all([this.loadReady(), this.loadLibrary(), this.loadHistory()]);
    this.readyTimer = window.setInterval(() => void this.loadReady(), 30000);
  }

  private showView(view: "research" | "ingest" | "resumes"): void {
    this.root.querySelector<HTMLElement>("#view-research")!.hidden = view !== "research";
    this.root.querySelector<HTMLElement>("#view-ingest")!.hidden = view !== "ingest";
    this.root.querySelector<HTMLElement>("#view-resumes")!.hidden = view !== "resumes";
    for (const b of Array.from(this.root.querySelectorAll<HTMLButtonElement>(".view-btn"))) {
      b.classList.toggle("active", b.dataset.view === view);
    }
    if (view === "ingest") void this.loadIngest();
    else void this.loadReady();
  }

  private setNarrowPane(pane: "query" | "reader"): void {
    const v = this.root.querySelector<HTMLElement>("#view-research");
    if (!v) return;
    v.classList.toggle("show-query", pane === "query");
    v.classList.toggle("show-reader", pane === "reader");
    if (pane === "reader") void this.reader.refresh();
  }

  private showReaderPane(): void {
    if (!window.matchMedia("(min-width: 980px)").matches) this.setNarrowPane("reader");
  }

  // --- readiness -----------------------------------------------------------------

  private async loadReady(): Promise<void> {
    try {
      const r = await api.ready();
      this.setReady("qdrant", r.qdrant);
      this.setReady("embedding", r.embedding_model);
      this.setReady("answer", r.answer_model);
      this.tokenRequired = r.token_required;
      if (!r.token_required) {
        // The server does not enforce a token: the token widget is
        // meaningless — keep it hidden and forget any stale stored token
        // so it stops being sent. (Re-poll every 30 s, so a server whose
        // config flips back on re-enables the field without a reload.)
        this.root.querySelector<HTMLElement>(".token")!.hidden = true;
        if (getToken()) setToken("");
      }
    } catch {
      this.setReady("qdrant", false);
      this.setReady("embedding", false);
      this.setReady("answer", false);
    }
  }

  private setReady(name: string, up: boolean): void {
    const n = this.root.querySelector<HTMLElement>(`#ready-${name}`);
    if (!n) return;
    n.classList.toggle("up", up);
    n.classList.toggle("down", !up);
    n.title = `${name}: ${up ? "ok" : "unavailable"}`;
  }

  // --- token ----------------------------------------------------------------------

  private handleApiError(e: unknown): void {
    if (e instanceof ApiError && e.status === 401) this.showTokenPrompt();
  }

  private showTokenPrompt(): void {
    // The server has told us it does not enforce a token: never surface the
    // field + "Set" button, even if a stray 401 arrives.
    if (this.tokenRequired === false) return;
    this.root.querySelector<HTMLElement>(".token")!.hidden = false;
    if (!getToken()) this.root.querySelector<HTMLInputElement>("#token-input")!.focus();
  }

  private saveToken(): void {
    const inp = this.root.querySelector<HTMLInputElement>("#token-input")!;
    setToken(inp.value);
    this.root.querySelector<HTMLElement>(".token")!.hidden = true;
    this.manifests.clear();
    this.resolvedManifests.clear();
    void Promise.all([this.loadReady(), this.loadLibrary(), this.loadHistory()]);
  }

  // --- library / filter --------------------------------------------------------------

  private async loadLibrary(): Promise<void> {
    try {
      const { books } = await api.library();
      this.books = books.filter((b) => b.published);
      const sel = this.root.querySelector<HTMLSelectElement>("#book-filter")!;
      sel.innerHTML = "";
      sel.append(el("option", { value: "" }, "All books"));
      for (const b of this.books) {
        sel.append(el("option", { value: b.rev_id }, `${b.title} (${b.format})`));
      }
      if (this.bookFilter) {
        const still = this.books.find((b) => b.rev_id === this.bookFilter!.rev);
        if (still) sel.value = still.rev_id;
        else this.bookFilter = null;
      }
    } catch (e) {
      this.handleApiError(e);
      this.resultsError(errText(e));
    }
  }

  private onBookFilterChange(sel: HTMLSelectElement): void {
    const book = this.books.find((b) => b.rev_id === sel.value);
    this.bookFilter = book ? { doc: book.doc_id, rev: book.rev_id } : null;
  }

  private bookQ(): { doc?: string; rev?: string } {
    return this.bookFilter ? { doc: this.bookFilter.doc, rev: this.bookFilter.rev } : {};
  }

  private setMode(mode: "answer" | "search"): void {
    this.mode = mode;
    for (const b of Array.from(this.root.querySelectorAll<HTMLButtonElement>(".mode-btn"))) {
      b.classList.toggle("active", b.dataset.mode === mode);
    }
    const run = this.root.querySelector<HTMLButtonElement>("#run-btn")!;
    run.textContent = mode === "answer" ? "Ask" : "Search";
  }

  // --- run ---------------------------------------------------------------------------

  private async run(): Promise<void> {
    if (this.running) return;
    const query = (this.root.querySelector<HTMLTextAreaElement>("#query") as HTMLTextAreaElement).value.trim();
    if (query.length === 0) {
      this.resultsMsg("Type a query first.");
      return;
    }
    this.running = true;
    this.setRunLabel(this.mode === "answer" ? "Answering…" : "Searching…");
    this.root.querySelector<HTMLElement>("#results")!.setAttribute("aria-busy", "true");
    try {
      if (this.mode === "answer") {
        const row = await api.answer({ query, ...this.bookQ() });
        this.currentAnswer = row;
        this.renderAnswer(row);
        await this.loadHistory();
      } else {
        const resp = await api.search({ query, ...this.bookQ(), limit: 20 });
        this.renderSearch(resp);
      }
    } catch (e) {
      this.handleApiError(e);
      this.resultsError(errText(e));
    } finally {
      this.running = false;
      this.setRunLabel(this.mode === "answer" ? "Ask" : "Search");
      this.root.querySelector<HTMLElement>("#results")!.setAttribute("aria-busy", "false");
    }
  }

  private setRunLabel(label: string): void {
    const b = this.root.querySelector<HTMLButtonElement>("#run-btn")!;
    b.textContent = label;
    b.disabled = this.running;
  }

  // --- answer rendering ------------------------------------------------------------------

  private renderAnswer(row: AnswerRow): void {
    const results = this.root.querySelector<HTMLElement>("#results")!;
    results.innerHTML = "";
    const card = el("div", { class: `answer-card status-${row.status}` },
      el("div", { class: "answer-head" },
        el("span", { class: `pill pill-${row.status}` }, row.status),
        el("span", { class: "answer-q" }, row.query),
        el("span", { class: "spacer" }),
        el("span", { class: "muted small" }, this.answerMeta(row)),
      ));
    if (row.status === "answered" && row.answer_text) {
      card.append(el("div", { class: "answer-text" }, row.answer_text));
    }
    if (row.abstain_reason) {
      card.append(el("div", { class: "answer-reason" }, `Abstained: ${row.abstain_reason}`));
    }
    if (row.failure_reason) {
      card.append(el("div", { class: "answer-reason err" }, `Failed: ${row.failure_reason}`));
    }
    if (row.evidence.length > 0) {
      const chips = el("div", { class: "cite-chips" });
      row.evidence.forEach((ev, i) => {
        chips.append(button(`[${i + 1}]`, "btn cite-chip", () => this.locateEvidence(ev)));
      });
      card.append(el("div", { class: "cite-label" }, "Citations"), chips);
      const list = el("div", { class: "evidence-list" });
      row.evidence.forEach((ev, i) => list.append(this.evidenceItem(ev, i)));
      card.append(list);
    }
    results.append(card);
    results.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  private answerMeta(row: AnswerRow): string {
    const parts: string[] = [];
    if (row.retrieval_ms !== null) parts.push(`retrieval ${row.retrieval_ms} ms`);
    if (row.model_ms !== null) parts.push(`model ${row.model_ms} ms`);
    if (row.model_revision) parts.push(row.model_revision);
    if (row.prompt_version) parts.push(`prompt ${row.prompt_version}`);
    return parts.join(" · ");
  }

  private evidenceItem(ev: Evidence, i: number): HTMLElement {
    const item = el("div", { class: "evidence" },
      el("div", { class: "evidence-head" },
        el("button", { class: "btn cite-chip" }, `[${i + 1}]`),
        el("span", { class: "evidence-src" }, ev.source_title),
        el("span", { class: "muted small" }, `${locationLabel(ev.location)} · ${ev.format}`),
        el("span", { class: "spacer" }),
        // Operator ask: open this book's stored summary (M8 resume) in the
        // Resumes view, next to the Google button.
        button("Summary", "btn small", () => {
          this.showView("resumes");
          void this.openResume(ev.rev_id, ev.source_title);
        }),
        // Operator ask: search the reference title in Google, far right of
        // the line (new tab; the title goes in the query verbatim).
        button("Google", "btn small", () => {
          window.open(
            `https://www.google.com/search?q=${encodeURIComponent(ev.source_title)}`,
            "_blank",
            "noopener,noreferrer",
          );
        }),
      ),
      el("div", { class: "evidence-text" }, ev.text),
    );
    item.querySelector<HTMLElement>(".cite-chip")!.addEventListener("click", () => {
      void this.locateEvidence(ev);
    });
    if (ev.quality_flags.length > 0) {
      item.append(el("div", { class: "flags" },
        ...ev.quality_flags.map((f) => el("span", { class: "flag" }, f))));
    }
    const excerpt = el("div", { class: "excerpt", hidden: "true" });
    item.append(
      el("div", { class: "evidence-actions" },
        button("Locate in reader", "btn small", () => this.locateEvidence(ev)),
        button("Excerpt", "btn small", () => this.loadExcerpt(excerpt, ev)),
      ),
      excerpt,
    );
    return item;
  }

  private async loadExcerpt(box: HTMLElement, ev: Evidence): Promise<void> {
    const row = this.currentAnswer;
    if (!row) return;
    box.hidden = false;
    box.textContent = "Resolving…";
    try {
      const c = await api.citation(row.answer_id, ev.evidence_id);
      if (c.available && c.excerpt) {
        box.textContent = c.excerpt.length > 400 ? `${c.excerpt.slice(0, 400)}…` : c.excerpt;
      } else {
        box.textContent = `Excerpt unavailable: ${c.reason ?? "no reason given"}`;
      }
    } catch (e) {
      box.textContent = errText(e);
    }
  }

  // --- search rendering ----------------------------------------------------------------------

  private renderSearch(resp: SearchResponse): void {
    const results = this.root.querySelector<HTMLElement>("#results")!;
    results.innerHTML = "";
    if (resp.degraded) {
      results.append(el("div", { class: "banner warn" },
        `Degraded results: ${resp.degraded_reason ?? "unknown reason"}`));
    }
    const counts = Object.entries(resp.counts)
      .map(([k, v]) => `${k}=${v}`)
      .join("  ");
    results.append(el("div", { class: "muted small search-meta" },
      `${resp.passages.length} passage(s)${counts ? ` · ${counts}` : ""}`));
    if (resp.passages.length === 0) {
      results.append(el("div", { class: "muted" }, "No passages found."));
      return;
    }
    const list = el("div", { class: "passages" });
    resp.passages.forEach((p, i) => list.append(this.passageItem(p, i)));
    results.append(list);
  }

  private passageItem(p: Passage, i: number): HTMLElement {
    const item = el("div", { class: "passage" },
      el("div", { class: "passage-head" },
        el("span", { class: "muted small" }, `#${i + 1}`),
        p.title ? el("span", { class: "passage-title" }, p.title) : el("span", { class: "passage-title" }, p.doc_id),
        el("span", { class: "muted small" }, `score ${p.score.toFixed(3)}`),
        el("span", { class: "muted small" },
          `dense ${p.dense_rank ?? "–"} · sparse ${p.sparse_rank ?? "–"}`),
      ),
      el("div", { class: "passage-text" }, p.text),
    );
    if (p.spans.length > 0) {
      const acts = el("div", { class: "evidence-actions" });
      p.spans.forEach((s, j) => {
        acts.append(
          button(p.spans.length > 1 ? `Locate ${j + 1}` : "Locate in reader", "btn small",
            () => this.locateSpan(p, s)));
      });
      item.append(acts);
    }
    return item;
  }

  // --- history ---------------------------------------------------------------------------------

  private async loadHistory(): Promise<void> {
    try {
      const { answers } = await api.answers(20);
      const ul = this.root.querySelector<HTMLUListElement>("#history")!;
      ul.innerHTML = "";
      if (answers.length === 0) {
        ul.append(el("li", { class: "muted" }, "No answers yet."));
        return;
      }
      for (const a of answers) {
        const item = el("li", { class: "hist-item" },
          el("span", { class: `pill pill-${a.status}` }, a.status),
          el("span", { class: "hist-q" }, a.query),
          el("span", { class: "hist-meta" }, `${a.citation_count} cit. · ${timeAgo(a.created_at)}`));
        item.addEventListener("click", () => void this.openHistoryAnswer(a.answer_id));
        ul.append(item);
      }
    } catch (e) {
      this.handleApiError(e);
    }
  }

  private async openHistoryAnswer(id: string): Promise<void> {
    try {
      const row = await api.oneAnswer(id);
      this.currentAnswer = row;
      this.renderAnswer(row);
    } catch (e) {
      this.readerStatus(errText(e));
    }
  }

  // --- locating in the reader ---------------------------------------------------------------------

  private manifestFor(rev: string): Promise<BookManifest | null> {
    const hit = this.manifests.get(rev);
    if (hit) return hit;
    const p: Promise<BookManifest | null> = api.book(rev)
      .then((m) => {
        this.resolvedManifests.set(rev, m);
        return m;
      })
      .catch((e): BookManifest | null => {
        this.readerStatus(errText(e));
        return null;
      });
    this.manifests.set(rev, p);
    return p;
  }

  /** Make sure the reader shows something from *rev* (page 1 / first section). */
  private async openReaderFor(rev: string): Promise<boolean> {
    const m = await this.manifestFor(rev);
    if (!m) return false;
    const s = this.reader.state();
    if (s.revId === rev && s.kind !== "unavailable") return true;
    const first = m.units[0];
    if (!first) {
      this.readerStatus("This book has no readable units yet.");
      return false;
    }
    if (first.kind === "section") await this.reader.openSection(rev, first);
    else await this.reader.openPdf(rev, 1);
    return this.reader.state().kind !== "unavailable";
  }

  private async locateEvidence(ev: Evidence): Promise<void> {
    this.showReaderPane();
    const m = await this.manifestFor(ev.rev_id);
    if (!m) return;
    const loc = ev.location;
    if (loc.kind === "section") {
      const unit = m.units.find((u) => u.kind === "section" && u.ref !== null && u.ref === loc.ref);
      if (!unit) {
        this.readerStatus("Section not found in this revision's manifest.");
        return;
      }
      await this.reader.openSection(ev.rev_id, unit, loc.anchor ?? null);
      return;
    }
    if (loc.kind === "page") {
      if (!(await this.openReaderFor(ev.rev_id))) return;
      if (this.reader.state().kind !== "pdf") {
        this.readerStatus("This book is not a PDF, so the page cannot be shown.");
        return;
      }
      if (ev.boxes.length > 0) await this.reader.goToEvidence(ev.boxes);
      else if (loc.page !== null && loc.page !== undefined) await this.reader.showPdfPage(loc.page);
      else this.readerStatus("On the located page — no precise box was recorded.");
      return;
    }
    this.readerStatus("This evidence has no locatable position.");
  }

  private async locateSpan(p: Passage, span: PassageSpan): Promise<void> {
    this.showReaderPane();
    const m = await this.manifestFor(p.rev_id);
    if (!m) return;
    const unit = m.units.find((u) => u.unit_id === span.unit_id);
    if (!unit) {
      this.readerStatus("Source unit is not in this revision's manifest.");
      return;
    }
    if (unit.kind === "section") {
      await this.reader.openSection(p.rev_id, unit);
      return;
    }
    if (!(await this.openReaderFor(p.rev_id))) return;
    await this.reader.goToEvidence([
      { unit_id: span.unit_id, page: unit.position + 1, bbox: span.bbox },
    ]);
  }

  // --- reader pane plumbing --------------------------------------------------------------------------

  private onReaderState(s: ReaderState): void {
    const show = (sel: string, on: boolean): void => {
      const n = this.root.querySelector<HTMLElement>(sel);
      if (n) n.hidden = !on;
    };
    show("#pdf-wrap", s.kind === "pdf");
    show("#epub-host", s.kind === "section");
    show("#reader-unavail", s.kind === "unavailable");
    show("#reader-empty", s.kind === "empty");
    const unavail = this.root.querySelector<HTMLElement>("#reader-unavail");
    if (unavail) unavail.textContent = s.unavailable ?? "";

    let pos = "—";
    if (s.kind === "pdf") {
      pos = `Page ${s.page} / ${s.pageCount}` + (s.pageLabel ? ` (${s.pageLabel})` : "");
    } else if (s.kind === "section") {
      pos = s.title ?? s.unitRef ?? "Section";
      if (s.revId && s.unitId) {
        const m = this.resolvedManifests.get(s.revId);
        const sections = m ? m.units.filter((u) => u.kind === "section") : null;
        if (sections) {
          const i = sections.findIndex((u) => u.unit_id === s.unitId);
          if (i >= 0) pos = `Section ${i + 1} / ${sections.length}` + (s.title ? ` — ${s.title}` : "");
        }
      }
    }
    this.set("r-pos", pos);
    // Source path line: every locate flow awaits the manifest before
    // changing reader state, so the manifest for s.revId is resolved here
    // whenever the server sent one (services.show_path_to_original).
    const path = s.revId ? this.resolvedManifests.get(s.revId)?.source_path : undefined;
    const pathEl = this.root.querySelector<HTMLElement>("#r-path");
    if (pathEl) {
      if (path) {
        pathEl.textContent = path;
        pathEl.hidden = false;
      } else {
        pathEl.hidden = true;
      }
    }
    const navigable = s.kind === "pdf" || s.kind === "section";
    this.disable("#r-prev", !navigable);
    this.disable("#r-next", !navigable);
  }

  private async readerStep(dir: -1 | 1): Promise<void> {
    const s = this.reader.state();
    if (s.kind === "pdf") {
      await this.reader.showPdfPage(s.page + dir);
      return;
    }
    if (s.kind !== "section" || !s.revId || !s.unitId) return;
    const m = this.resolvedManifests.get(s.revId);
    if (!m) return;
    const sections = m.units.filter((u) => u.kind === "section");
    const i = sections.findIndex((u) => u.unit_id === s.unitId);
    const next = i >= 0 ? sections[i + dir] : undefined;
    if (next) await this.reader.openSection(s.revId, next);
  }

  private async downloadCurrent(): Promise<void> {
    const s = this.reader.state();
    if (s.revId === null) {
      this.readerStatus("No book open in the reader.");
      return;
    }
    const m = this.resolvedManifests.get(s.revId);
    const book = m ? this.books.find((b) => b.rev_id === s.revId) : undefined;
    const base =
      (book?.title || `book-${s.revId}`)
        .replace(/[^\w.-]+/g, "-")
        .replace(/-+$/, "")
        .slice(0, 80) || "book";
    const ext = (m?.format ?? book?.format ?? "pdf").toLowerCase();
    this.readerStatus(`Downloading ${base}.${ext} …`);
    try {
      await downloadSource(s.revId, `${base}.${ext}`);
      this.readerStatus(`Downloaded ${base}.${ext}`);
    } catch (e) {
      this.readerStatus(errText(e));
    }
  }

  private readerStatus(msg: string): void {
    this.set("reader-status", msg);
  }

  // --- results helpers -----------------------------------------------------------------------------------

  private resultsMsg(msg: string): void {
    const r = this.root.querySelector<HTMLElement>("#results")!;
    const first = r.querySelector(".banner");
    if (!first) r.prepend(el("div", { class: "banner" }, msg));
    else first.textContent = msg;
  }

  private resultsError(msg: string): void {
    const r = this.root.querySelector<HTMLElement>("#results")!;
    const first = r.querySelector(".banner.err");
    if (!first) r.prepend(el("div", { class: "banner err" }, msg));
    else first.textContent = msg;
  }

  private set(id: string, text: string): void {
    const n = this.root.querySelector(`#${id}`);
    if (n) n.textContent = text;
  }

  private disable(sel: string, off: boolean): void {
    const n = this.root.querySelector<HTMLButtonElement>(sel);
    if (n) n.disabled = off;
  }

  // --- ingestion dashboard ---------------------------------------------------------------------------------

  private async loadIngest(): Promise<void> {
    try {
      const s = await api.ingestStatus();
      this.set("i-docs", String(s.documents));
      this.set("i-revs", String(s.revisions));
      this.set("i-chunks", String(s.chunks));
      this.set("i-answers", String(s.answers));
      this.set("i-schema", String(s.schema_version));
      this.set("i-paused", s.paused ? "PAUSED" : "running");
      const pausedCard = this.root.querySelector<HTMLElement>("#i-paused")!.closest(".card");
      if (pausedCard) pausedCard.classList.toggle("paused", s.paused);
      const jobs = this.root.querySelector<HTMLTableElement>("#i-jobs")!;
      jobs.innerHTML = "";
      const entries = Object.entries(s.jobs);
      if (entries.length === 0) {
        jobs.append(el("tr", {}, el("td", { colspan: "2", class: "muted" }, "no jobs")));
      } else {
        for (const [name, count] of entries) {
          jobs.append(el("tr", {}, el("td", {}, name), el("td", { class: "num" }, String(count))));
        }
      }
    } catch (e) {
      this.handleApiError(e);
      this.set("i-msg", errText(e));
    }
  }

  private async ingestScan(): Promise<void> {
    try {
      const { reports } = await api.scan();
      this.root.querySelector<HTMLElement>("#i-scan-report")!.textContent = JSON.stringify(reports, null, 2);
      const roots = reports.map((r) => `${r.root}: +${r.new_documents} doc, +${r.new_revisions} rev, ${r.jobs_enqueued} job(s)`);
      this.set("i-msg", `Scan complete — ${roots.length} root(s): ${roots.join(" | ")}`);
      await this.loadIngest();
    } catch (e) {
      this.handleApiError(e);
      this.set("i-msg", errText(e));
    }
  }

  private async ingestPause(): Promise<void> {
    try {
      const r = await api.ingestPause("paused from the research UI");
      this.set("i-msg", r.paused ? "Ingestion paused." : "Ingestion is not paused.");
      await this.loadIngest();
    } catch (e) {
      this.handleApiError(e);
      this.set("i-msg", errText(e));
    }
  }

  private async ingestResume(): Promise<void> {
    try {
      await api.ingestResume();
      this.set("i-msg", "Ingestion resumed.");
      await this.loadIngest();
    } catch (e) {
      this.handleApiError(e);
      this.set("i-msg", errText(e));
    }
  }

  private async ingestRetry(includePermanent: boolean): Promise<void> {
    try {
      const r = await api.ingestRetry(includePermanent);
      this.set("i-msg", `Requeued ${r.requeued} job(s).`);
      await this.loadIngest();
    } catch (e) {
      this.handleApiError(e);
      this.set("i-msg", errText(e));
    }
  }

  // --- resumes (M8) ------------------------------------------------------------------------

  private async resumeSearch(): Promise<void> {
    const query = this.root.querySelector<HTMLInputElement>("#resume-query")!.value.trim();
    if (!query) {
      this.set("resume-msg", "Type a keyword first.");
      return;
    }
    try {
      const { results } = await api.searchResumes(query);
      const box = this.root.querySelector<HTMLElement>("#resume-results")!;
      box.innerHTML = "";
      if (results.length === 0) {
        box.append(el("div", { class: "banner" }, "No stored resume mentions that keyword yet."));
        this.set("resume-msg", "0 matches.");
        return;
      }
      results.forEach((r, i) => box.append(this.resumeItem(r, i + 1)));
      this.set("resume-msg", `${results.length} match(es) — ranked by keyword relevance.`);
    } catch (e) {
      this.handleApiError(e);
      this.set("resume-msg", errText(e));
    }
  }

  private resumeItem(r: ResumeSummary, rank: number): HTMLElement {
    const item = el("div", { class: "passage resume-hit" },
      el("div", { class: "passage-head" },
        el("span", { class: "muted" }, `#${rank}`),
        el("span", { class: "passage-title" }, r.title),
        el("span", { class: "muted" }, `score ${r.score.toFixed(3)}`),
      ),
      el("div", { class: "passage-text" }, r.excerpt),
    );
    item.addEventListener("click", () => void this.openResume(r.rev_id, r.title));
    return item;
  }

  private async openResume(revId: string, fallbackTitle: string): Promise<void> {
    try {
      const rec = await api.resume(revId);
      this.resumeState = { rev: rec.rev_id, title: rec.title ?? fallbackTitle };
      this.set("resume-title", this.resumeState.title);
      this.set("resume-meta", `${rec.word_count} words · ${rec.model_revision} · ${rec.prompt_version}`);
      const text = this.root.querySelector<HTMLElement>("#resume-text")!;
      text.textContent = rec.text;
      text.classList.remove("muted");
      text.scrollTop = 0;
    } catch (e) {
      this.handleApiError(e);
      // A missing resume (HTTP 404) is normal while the backfill runs: say
      // so plainly — in the pane the user is looking at, not just in the
      // small search-status line — and never leave the previous book's meta
      // claiming a word count over an empty text area.
      const missing = e instanceof ApiError && e.status === 404;
      const msg = missing
        ? "No summary stored for this book yet — the résumé backfill is " +
          "still running. Use Open book to read the book itself in the meantime."
        : errText(e);
      // Still point resumeState at the clicked book: it exists even when its
      // resume has not been generated yet, so "Open book" must open *this*
      // book, not the previously loaded one (or nothing).
      this.resumeState = { rev: revId, title: fallbackTitle };
      this.set("resume-title", fallbackTitle);
      this.set("resume-meta", "");
      this.set("resume-msg", msg);
      const text = this.root.querySelector<HTMLElement>("#resume-text")!;
      text.textContent = msg;
      text.classList.toggle("muted", missing);
      text.scrollTop = 0;
    }
  }

  private async openResumeBook(): Promise<void> {
    const st = this.resumeState;
    if (!st) {
      this.set("resume-msg", "Pick a resume from the list first.");
      return;
    }
    this.showView("research");
    this.showReaderPane();
    // Opening a PDF fetches the whole archived source first, which can take
    // a few seconds — say we are on it so the click never looks dead.
    // openReaderFor's specific failure messages overwrite this.
    this.readerStatus(`Opening "${st.title}" …`);
    if (await this.openReaderFor(st.rev)) void this.reader.refresh();
  }
}
