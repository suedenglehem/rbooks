/**
 * Reader pane (PRD §10):
 *
 *  - PDF: pdf.js navigates by *physical* page; evidence boxes are fitz bboxes
 *    in page points (bottom-left origin, unrotated page space), transformed
 *    with the viewport so zoom and page rotation are both accounted for.
 *  - EPUB: sections render the server-sanitized HTML fragment through a second
 *    client-side sanitize pass, and deterministic paragraph anchors
 *    (`paragraphs[i].anchor`) are re-attached to the i-th rendered block-level
 *    element — mirroring extraction.parse_blocks' document-order walk.
 *
 * A missing archived original (410) or any other failure shows an explicit
 * unavailable message in the pane; it never substitutes a different edition.
 */
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
import type { PDFDocumentProxy, PDFPageProxy, RenderTask } from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { api, errText } from "./api";
import type { EvidenceBox, UnitRef, UnitPayload } from "./api";
import { sanitizeFragment } from "./sanitize";

// Vite emits the worker as a same-origin asset → satisfies `worker-src 'self'`.
GlobalWorkerOptions.workerSrc = workerUrl;

// Must mirror extraction.sanitize._BLOCK_TAGS (document order, non-empty text).
const BLOCK_SELECTOR = [
  "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre",
  "dt", "dd", "figcaption",
].join(",");

/** A four-element numeric rect, or null when the input is missing/invalid. */
function rect4(v: number[] | null | undefined): [number, number, number, number] | null {
  if (!v || v.length < 4) return null;
  const a = v[0], b = v[1], c = v[2], d = v[3];
  if (a === undefined || b === undefined || c === undefined || d === undefined) return null;
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || !Number.isFinite(d)) {
    return null;
  }
  return [a, b, c, d];
}

/**
 * Assign `paragraphs[i].anchor` as the id of the i-th rendered block-level
 * element (same tag set, document order, non-empty text) that the server's
 * parse_blocks would have counted. Returns the number of anchors assigned.
 */
export function assignAnchors(
  root: HTMLElement,
  paragraphs: { anchor: string; text: string }[],
): number {
  let i = 0;
  for (const el of Array.from(root.querySelectorAll(BLOCK_SELECTOR))) {
    if (!(el instanceof HTMLElement)) continue;
    if ((el.textContent ?? "").trim() === "") continue;
    const p = paragraphs[i];
    if (p) el.id = p.anchor;
    i += 1;
  }
  return i;
}

export interface ReaderState {
  kind: "empty" | "pdf" | "section" | "unavailable";
  revId: string | null;
  page: number; // pdf: current physical page (1-based)
  pageCount: number;
  pageLabel: string | null; // pdf: pdf.js label for the current page
  unitId: string | null; // section: current unit
  unitRef: string | null;
  title: string | null; // section: unit title
  unavailable: string | null;
}

const EMPTY_STATE: ReaderState = {
  kind: "empty",
  revId: null,
  page: 0,
  pageCount: 0,
  pageLabel: null,
  unitId: null,
  unitRef: null,
  title: null,
  unavailable: null,
};

function unavailableText(e: unknown): string {
  if (e instanceof Error && e.name === "ApiError" && "status" in e) {
    const status = (e as { status: number }).status;
    if (status === 410) {
      return (
        "The archived original for this revision is no longer available. " +
        "The reader cannot open a substitute edition."
      );
    }
    if (status === 409) return "This revision has not been extracted yet — no source to read.";
    if (status === 404) return "Unknown revision — the reader cannot open it.";
    if (status === 401) return "Authentication required — set the API token to read sources.";
  }
  return `Reader unavailable: ${errText(e)}`;
}

export class Reader {
  private doc: PDFDocumentProxy | null = null;
  private docRev: string | null = null;
  private pdfPage = 1;
  private pdfCount = 0;
  private pdfLabel: string | null = null;
  private boxes: EvidenceBox[] = [];
  private seq = 0;
  private renderTask: RenderTask | null = null;
  private sectionRev: string | null = null;
  private sectionUnit: UnitRef | null = null;
  private sectionTitle: string | null = null;
  private unavailableMsg: string | null = null;
  private zoom = 1; // multiplier on "fit width"
  private destroyed = false;

  constructor(
    private canvas: HTMLCanvasElement,
    private sectionHost: HTMLElement,
    private scrollHost: HTMLElement,
    private onState: (s: ReaderState) => void,
  ) {}

  // --- state ---------------------------------------------------------------

  state(): ReaderState {
    if (this.unavailableMsg !== null) {
      return { ...EMPTY_STATE, kind: "unavailable", revId: this.docRev ?? this.sectionRev, unavailable: this.unavailableMsg };
    }
    if (this.doc !== null) {
      return {
        ...EMPTY_STATE,
        kind: "pdf",
        revId: this.docRev,
        page: this.pdfPage,
        pageCount: this.pdfCount,
        pageLabel: this.pdfLabel,
      };
    }
    if (this.sectionUnit !== null) {
      return {
        ...EMPTY_STATE,
        kind: "section",
        revId: this.sectionRev,
        unitId: this.sectionUnit.unit_id,
        unitRef: this.sectionUnit.ref,
        title: this.sectionTitle,
      };
    }
    return { ...EMPTY_STATE };
  }

  private emit(): void {
    this.onState(this.state());
  }

  // --- zoom ----------------------------------------------------------------

  getZoom(): number {
    return this.zoom;
  }

  zoomBy(factor: number): void {
    this.zoom = Math.min(5, Math.max(0.25, this.zoom * factor));
    void this.refresh();
  }

  zoomFit(): void {
    this.zoom = 1;
    void this.refresh();
  }

  /** Re-render whatever is on screen (zoom/resize/pane-shown changes). */
  async refresh(): Promise<void> {
    if (this.doc !== null) {
      await this.showPdfPage(this.pdfPage);
    } else if (this.sectionUnit !== null) {
      this.applySectionZoom();
    }
  }

  private applySectionZoom(): void {
    this.sectionHost.style.setProperty("zoom", String(this.zoom));
  }

  // --- PDF -----------------------------------------------------------------

  /**
   * Open (or reuse) the PDF for *rev* and show *page* (1-based), optionally
   * with evidence boxes to highlight. A failed source load leaves the pane in
   * an explicit unavailable state.
   */
  async openPdf(rev: string, page = 1, boxes: EvidenceBox[] = []): Promise<void> {
    if (this.docRev === rev && this.doc !== null) {
      this.boxes = boxes;
      await this.showPdfPage(page);
      return;
    }
    const seq = ++this.seq;
    this.unavailableMsg = null;
    this.closeSection();
    const previous = this.doc;
    try {
      // pdf.js's internal Range requests cannot carry an Authorization
      // header, so the bytes are fetched by the authed client and handed over.
      const data = await api.sourceBytes(rev);
      if (this.destroyed || seq !== this.seq) return;
      this.doc = null;
      this.docRev = null;
      const doc = await getDocument({ data }).promise;
      if (this.destroyed || seq !== this.seq) {
        void doc.destroy();
        return;
      }
      this.doc = doc;
      this.docRev = rev;
      this.pdfCount = doc.numPages;
      this.pdfPage = Math.min(Math.max(1, page), doc.numPages);
      this.boxes = boxes;
      await this.showPdfPage(this.pdfPage);
      void previous?.destroy();
    } catch (e) {
      if (this.destroyed || seq !== this.seq) return;
      this.doc = null;
      this.docRev = null;
      this.pdfCount = 0;
      this.unavailableMsg = unavailableText(e);
      this.emit();
    }
  }

  /** PDF only. Navigate to the first box's page, then highlight all boxes. */
  async goToEvidence(boxes: EvidenceBox[]): Promise<void> {
    if (this.doc === null || this.docRev === null) return;
    this.boxes = boxes;
    let target = this.pdfPage;
    for (const b of boxes) {
      if (typeof b.page === "number" && b.page >= 1) {
        target = b.page;
        break;
      }
    }
    await this.showPdfPage(target);
  }

  async showPdfPage(page: number): Promise<void> {
    const doc = this.doc;
    if (doc === null) return;
    page = Math.min(Math.max(1, page), doc.numPages);
    const seq = ++this.seq;
    this.pdfPage = page;
    this.emit();
    try {
      const pdfPage = await doc.getPage(page);
      if (this.destroyed || seq !== this.seq || this.doc !== doc) return;
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const vp1 = pdfPage.getViewport({ scale: 1 });
      const avail = Math.max(240, this.scrollHost.clientWidth - 32);
      const base = Math.min(avail / vp1.width, 4);
      const scale = base * this.zoom * dpr;
      const vp = pdfPage.getViewport({ scale });
      this.canvas.width = Math.floor(vp.width);
      this.canvas.height = Math.floor(vp.height);
      this.canvas.style.width = `${Math.floor(vp.width / dpr)}px`;
      this.canvas.style.height = `${Math.floor(vp.height / dpr)}px`;
      const ctx = this.canvas.getContext("2d");
      if (ctx === null) return;
      if (this.renderTask !== null) {
        try {
          this.renderTask.cancel();
        } catch {
          // already finished
        }
        this.renderTask = null;
      }
      const labels = await doc.getPageLabels();
      const label = labels?.[page - 1];
      this.pdfLabel = label && label !== "" ? label : null;
      const task = pdfPage.render({ canvas: this.canvas, viewport: vp });
      this.renderTask = task;
      try {
        await task.promise;
      } catch (e) {
        this.renderTask = null;
        if (e instanceof Error && e.name === "RenderingCancelledException") return;
        throw e;
      }
      this.renderTask = null;
      if (this.destroyed || seq !== this.seq || this.doc !== doc) return;
      this.drawBoxes(pdfPage, vp, dpr);
      this.emit();
    } catch (e) {
      if (this.destroyed || seq !== this.seq) return;
      this.unavailableMsg = `PDF rendering failed: ${errText(e)}`;
      this.emit();
    }
  }

  /**
   * Draw the evidence boxes that belong to the current page.
   *
   * fitz bboxes are [x0, y0, x1, y1] in page points with a *bottom-left*
   * origin in *unrotated* page space. Convert to a top-left rectangle in the
   * same space, then let the (rotated, scaled) viewport do the rest.
   */
  private drawBoxes(page: PDFPageProxy, vp: { convertToViewportRectangle(r: number[]): number[] }, dpr: number): void {
    const ctx = this.canvas.getContext("2d");
    if (ctx === null) return;
    const vp1 = page.getViewport({ scale: 1 });
    const unrotated = (page.rotate + 180) % 360 === 0;
    const pageH = unrotated ? vp1.height : vp1.width; // unrotated height in pt
    ctx.save();
    for (const box of this.boxes) {
      if (box.page !== this.pdfPage) continue;
      const b = rect4(box.bbox);
      if (b === null) continue;
      const [x0, y0, x1, y1] = b;
      const w = x1 - x0;
      const h = y1 - y0;
      if (w <= 0 || h <= 0) continue;
      const r = rect4(vp.convertToViewportRectangle([x0, pageH - y1, w, h]));
      if (r === null) continue;
      ctx.fillStyle = "rgba(255, 170, 0, 0.22)";
      ctx.strokeStyle = "#d97706";
      ctx.lineWidth = 2 * dpr;
      ctx.fillRect(r[0], r[1], r[2], r[3]);
      ctx.strokeRect(r[0], r[1], r[2], r[3]);
    }
    ctx.restore();
  }

  private closePdf(): void {
    if (this.renderTask !== null) {
      try {
        this.renderTask.cancel();
      } catch {
        // already finished
      }
      this.renderTask = null;
    }
    const doc = this.doc;
    this.doc = null;
    this.docRev = null;
    this.pdfPage = 1;
    this.pdfCount = 0;
    this.pdfLabel = null;
    this.boxes = [];
    this.canvas.width = 0;
    this.canvas.height = 0;
    void doc?.destroy();
  }

  // --- EPUB sections ---------------------------------------------------------

  /** Render one section unit (with optional anchor scroll). */
  async openSection(rev: string, unit: UnitRef, anchor?: string | null): Promise<void> {
    const seq = ++this.seq;
    this.unavailableMsg = null;
    this.closePdf();
    this.sectionRev = rev;
    this.sectionUnit = unit;
    this.sectionTitle = null;
    this.sectionHost.innerHTML = "";
    this.emit();
    try {
      const payload: UnitPayload = await api.unit(rev, unit.unit_id);
      if (this.destroyed || seq !== this.seq) return;
      if (payload.kind !== "section") {
        throw new Error(`unit ${unit.unit_id} is not a section`);
      }
      const html = sanitizeFragment(payload.sanitized_html ?? "");
      this.sectionHost.innerHTML = html;
      const assigned = assignAnchors(this.sectionHost, payload.paragraphs ?? []);
      if (assigned < (payload.paragraphs ?? []).length) {
        // Not fatal: anchors are best-effort; the section still renders.
        console.warn(
          `anchor mismatch: ${assigned}/${payload.paragraphs?.length ?? 0} assigned for ${unit.unit_id}`,
        );
      }
      this.sectionTitle = payload.title ?? null;
      this.applySectionZoom();
      this.emit();
      if (anchor !== null && anchor !== undefined && seq === this.seq) {
        this.locateAnchor(anchor);
      }
    } catch (e) {
      if (this.destroyed || seq !== this.seq) return;
      this.sectionUnit = null;
      this.unavailableMsg = unavailableText(e);
      this.emit();
    }
  }

  /** Scroll to a deterministic paragraph anchor and flash it. */
  locateAnchor(anchor: string): boolean {
    const el = this.sectionHost.querySelector(`[id="${CSS.escape(anchor)}"]`);
    if (!(el instanceof HTMLElement)) return false;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    el.classList.add("anchor-flash");
    window.setTimeout(() => el.classList.remove("anchor-flash"), 1800);
    return true;
  }

  private closeSection(): void {
    this.sectionRev = null;
    this.sectionUnit = null;
    this.sectionTitle = null;
    this.sectionHost.innerHTML = "";
  }

  // --- lifecycle -------------------------------------------------------------

  destroy(): void {
    this.destroyed = true;
    this.seq += 1;
    this.closePdf();
    this.closeSection();
    this.unavailableMsg = null;
  }
}
