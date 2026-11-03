const { t, el, api, formatDate } = window.SC;
const $ = (id) => document.getElementById(id);

function message(kind, text) { return el("div", { class: `msg ${kind}` }, text); }

// Error text is looked up again on render, so a language switch translates old messages too.
function errorText(e) {
  if (e && e.code && window.SC.has(`ui.errors.${e.code}`)) return t(`ui.errors.${e.code}`);
  if (e && e.status) return t("ui.errors.status", { status: e.status });
  return t("ui.errors.unreachable");
}

// --- status and banner ----------------------------------------------------------
let health = null;     // last /health response
let offline = false;   // the last /health call failed

async function refreshStatus() {
  try {
    health = await api("/health");
    offline = false;
  } catch {
    offline = true;
  }
  renderStatus();
}

function renderStatus() {
  const status = $("status");
  if (offline) {
    status.replaceChildren(el("span", { class: "pill down" }, t("ui.status.offline")));
  } else if (health) {
    status.replaceChildren(
      el("span", { class: "pill" }, t("ui.status.documents", { n: health.documents })),
      el("span", { class: `pill ${health.llm_enabled ? "on" : "off"}` },
        t(health.llm_enabled ? "ui.status.llm_on" : "ui.status.llm_off")),
    );
  }

  const banner = $("banner");
  if (offline) {
    const retry = el("button", { class: "ghost", type: "button" }, t("ui.banner.retry"));
    retry.addEventListener("click", reloadAll);
    banner.replaceChildren(message("err", t("ui.banner.offline")), retry);
    banner.hidden = false;
  } else if (health && !health.llm_enabled) {
    banner.replaceChildren(message("warn", t("ui.banner.llm_off")));
    banner.hidden = false;
  } else {
    banner.replaceChildren();
    banner.hidden = true;
  }
}

// --- knowledge base ---------------------------------------------------------------
let documents = null;  // null until loaded, [] when empty
let documentsError = null;
let uploadLog = [];    // [{ kind, text: () => string }], re-rendered on language change

async function loadDocuments() {
  try {
    documents = await api("/kb");
    documentsError = null;
  } catch (e) {
    documentsError = e;
  }
  renderDocuments();
}

function renderDocuments() {
  const list = $("docs");
  if (documentsError && documents === null) {
    list.replaceChildren(el("li", { class: "empty" }, message("err", t("ui.kb.load_failed"))));
    return;
  }
  if (documents === null) return;
  if (!documents.length) {
    list.replaceChildren(el("li", { class: "empty" }, t("ui.kb.empty")));
    return;
  }
  list.replaceChildren(...documents.map((d) => el("li", {},
    el("span", { class: "name" }, d.filename),
    el("span", { class: "when" }, t("ui.kb.added_at", { date: formatDate(d.created_at) })),
  )));
}

function renderUploadLog() {
  $("uploadMsg").replaceChildren(...uploadLog.map((entry) => message(entry.kind, entry.text())));
}

async function uploadFiles(files) {
  uploadLog = [];
  for (const file of files) {
    const name = file.name;
    const entry = { kind: "info", text: () => t("ui.kb.uploading", { name }) };
    uploadLog.push(entry);
    renderUploadLog();
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await api("/kb/upload", { method: "POST", body: form });
      const skipped = r.skipped_pages || [];
      entry.kind = skipped.length ? "warn" : "ok";
      entry.text = () => {
        let text = t(r.duplicate ? "ui.kb.duplicate" : "ui.kb.added", { name, chunks: r.chunks });
        if (skipped.length) text += "; " + t("ui.kb.skipped_pages", { pages: skipped.join(", ") });
        return text;
      };
    } catch (e) {
      entry.kind = "err";
      entry.text = () => t("ui.kb.failed", { name, error: errorText(e) });
    }
    renderUploadLog();
  }
  loadDocuments();
  refreshStatus();
}

$("file").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  uploadFiles([...e.dataTransfer.files]);
});

// --- tickets: list and filter ----------------------------------------------------------
const STATUSES = ["new", "drafted", "escalated", "resolved"];
let tickets = null;    // all tickets, newest first; null until loaded
let ticketsError = null;
let filter = window.SC.store.get("filter");
if (!STATUSES.includes(filter)) filter = "all";
let selectedId = null;

function statusBadge(status) {
  const known = STATUSES.includes(status);
  return el("span", { class: `badge s-${known ? status : "new"}` }, known ? t(`ui.tickets.status_${status}`) : status);
}

// One request for all tickets: the filter chips show a count per status, and filtering stays instant.
async function loadTickets() {
  try {
    tickets = (await api("/tickets")).reverse();
    ticketsError = null;
  } catch (e) {
    ticketsError = e;
  }
  renderFilters();
  renderTickets();
}

function renderFilters() {
  const counts = { all: tickets ? tickets.length : 0 };
  for (const s of STATUSES) counts[s] = tickets ? tickets.filter((x) => x.status === s).length : 0;
  $("filters").replaceChildren(...["all", ...STATUSES].map((key) => {
    const chip = el("button", { class: "chip", type: "button", "aria-pressed": String(filter === key) },
      t(`ui.tickets.filter_${key}`), el("span", { class: "n" }, tickets ? String(counts[key]) : ""));
    chip.addEventListener("click", () => {
      filter = key;
      window.SC.store.set("filter", key);
      renderFilters();
      renderTickets();
    });
    return chip;
  }));
}

function renderTickets() {
  const list = $("ticketList");
  $("ticketsMsg").replaceChildren(
    ticketsError ? message("err", `${t("ui.tickets.load_failed")} ${errorText(ticketsError)}`) : "");
  if (tickets === null) { list.replaceChildren(); return; }
  if (!tickets.length) {
    list.replaceChildren(el("li", { class: "empty" }, t("ui.tickets.empty")));
    $("newTicket").open = true;
    return;
  }
  const shown = filter === "all" ? tickets : tickets.filter((x) => x.status === filter);
  if (!shown.length) {
    list.replaceChildren(el("li", { class: "empty" }, t("ui.tickets.empty_filter")));
    return;
  }
  list.replaceChildren(...shown.map((x) => {
    const button = el("button", {
      class: "ticket", type: "button", "aria-current": String(x.id === selectedId),
      "aria-label": t("ui.tickets.open", { id: x.id, name: x.customer_name }),
    },
      el("div", { class: "ticket-top" },
        el("span", { class: "ticket-id" }, `#${x.id}`),
        el("span", { class: "ticket-name" }, x.customer_name),
        statusBadge(x.status)),
      el("p", { class: "ticket-q" }, x.question),
      el("div", { class: "ticket-when" }, formatDate(x.created_at)),
    );
    button.addEventListener("click", () => selectTicket(x.id, { scroll: true }));
    return el("li", {}, button);
  }));
}

// Keep the list in sync after the card changed a ticket, without refetching everything.
function updateTicketInList(ticket) {
  if (!tickets) return;
  const i = tickets.findIndex((x) => x.id === ticket.id);
  if (i === -1) tickets.unshift(ticket);
  else tickets[i] = { ...tickets[i], ...ticket };
  renderFilters();
  renderTickets();
}

// --- tickets: create ------------------------------------------------------------------
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const FIELDS = {
  customer_name: { input: "tName", error: "ui.tickets.name_required" },
  customer_email: { input: "tEmail", error: "ui.tickets.email_invalid" },
  question: { input: "tQuestion", error: "ui.tickets.question_required" },
};
let formErrors = [];   // field names, re-rendered on language change
let formNotice = null; // { kind, text: () => string }

function renderFormMessages() {
  for (const [name, field] of Object.entries(FIELDS)) {
    $(field.input).closest(".field").classList.toggle("invalid", formErrors.includes(name));
    $(field.input).setAttribute("aria-invalid", String(formErrors.includes(name)));
  }
  const lines = formErrors.map((name) => message("err", t(FIELDS[name].error)));
  if (formNotice) lines.push(message(formNotice.kind, formNotice.text()));
  $("ticketFormMsg").replaceChildren(...lines);
}

async function createTicket() {
  const body = {
    customer_name: $("tName").value.trim(),
    customer_email: $("tEmail").value.trim(),
    question: $("tQuestion").value.trim(),
  };
  formErrors = [];
  formNotice = null;
  if (!body.customer_name) formErrors.push("customer_name");
  if (!EMAIL.test(body.customer_email)) formErrors.push("customer_email");
  if (!body.question) formErrors.push("question");
  renderFormMessages();
  if (formErrors.length) { $(FIELDS[formErrors[0]].input).focus(); return; }

  const button = $("createTicket");
  button.disabled = true;
  button.classList.add("busy");
  button.textContent = t("ui.tickets.creating");
  try {
    const ticket = await api("/tickets", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    $("ticketForm").reset();
    formNotice = { kind: "ok", text: () => t("ui.tickets.created", { id: ticket.id }) };
    // Show the new ticket even when the current filter would hide it.
    if (filter !== "all" && filter !== ticket.status) { filter = "all"; window.SC.store.set("filter", "all"); }
    updateTicketInList(ticket);
    selectTicket(ticket.id);
  } catch (e) {
    // A 422 names the rejected fields; anything else is a general message.
    const fields = (e.detail || []).map((d) => (d.loc || []).at(-1)).filter((name) => name in FIELDS);
    if (fields.length) formErrors = [...new Set(fields)];
    else formNotice = { kind: "err", text: () => errorText(e) };
  } finally {
    button.disabled = false;
    button.classList.remove("busy");
    button.textContent = t("ui.tickets.create");
    renderFormMessages();
  }
}

$("ticketForm").addEventListener("submit", (e) => { e.preventDefault(); createTicket(); });
$("refreshTickets").addEventListener("click", () => { loadTickets(); refreshStatus(); });

// --- ticket card ------------------------------------------------------------------------
let ticketDetail = null; // GET /tickets/{id} of the selected ticket
let ticketError = null;

async function selectTicket(id, { scroll = false } = {}) {
  if (selectedId !== id) {
    confirmingResolve = false;
    cardNotice = null;
  }
  selectedId = id;
  history.replaceState(null, "", `#ticket-${id}`);
  renderTickets();
  if (!ticketDetail || ticketDetail.id !== id) ticketDetail = null;
  ticketError = null;
  renderTicket();
  // One column on phones: the card is below the list, bring it into view.
  if (scroll && matchMedia("(max-width: 899px)").matches) {
    $("ticket").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  try {
    const detail = await api(`/tickets/${id}`);
    if (selectedId !== id) return; // the operator already picked another ticket
    ticketDetail = detail;
    updateTicketInList(detail);
  } catch (e) {
    if (selectedId !== id) return;
    ticketError = e;
  }
  renderTicket();
}

function backToList() {
  $("tickets").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTicket() {
  const body = $("ticketBody");
  $("ticketHeading").textContent = selectedId ? `${t("ui.ticket.heading")} #${selectedId}` : t("ui.ticket.heading");
  if (!selectedId) {
    body.replaceChildren(el("p", { class: "placeholder" }, t("ui.ticket.none")));
    return;
  }
  const back = el("button", { class: "ghost back", type: "button" }, t("ui.ticket.back"));
  back.addEventListener("click", backToList);
  if (ticketError) {
    body.replaceChildren(back, message("err", errorText(ticketError)));
    return;
  }
  // Until the detail arrives, show what the list already knows.
  const ticket = ticketDetail || (tickets || []).find((x) => x.id === selectedId);
  if (!ticket) {
    body.replaceChildren(back, el("p", { class: "placeholder" }, el("span", { class: "spinner" })));
    return;
  }
  const parts = [
    back,
    el("div", { class: "ticket-meta" },
      el("strong", {}, ticket.customer_name),
      el("span", { class: "email" }, ticket.customer_email),
      el("span", {}, t("ui.ticket.created_at", { date: formatDate(ticket.created_at) })),
      statusBadge(ticket.status)),
    el("div", { class: "label" }, t("ui.ticket.customer_question")),
    el("p", { class: "question" }, ticket.question),
  ];
  if (ticketDetail && ticketDetail.id === ticket.id) parts.push(...draftSection(ticketDetail));
  body.replaceChildren(...parts);
}

// --- ticket card: draft, sources, escalation ----------------------------------------------
const REASONS = ["no_answer_in_kb", "no_relevant_documents", "no_valid_citations", "llm_unavailable", "embedding_unavailable"];
const drafting = new Set();  // ticket ids with a draft request in flight
let cardNotice = null;       // { id, kind, text: () => string } for the selected ticket
let confirmingResolve = false;

function reasonText(code) {
  return t(`ui.reasons.${REASONS.includes(code) ? code : "unknown"}`);
}

function latestDraft(ticket) {
  return ticket.drafts.length ? ticket.drafts[ticket.drafts.length - 1] : null;
}

// The reply the operator would send: the latest draft that is an answer, not an escalation note.
function replyDraft(ticket) {
  const draft = latestDraft(ticket);
  return draft && draft.confidence === "answered" ? draft : null;
}

function jumpTo(n) {
  const target = $(`src-${n}`);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  target.classList.remove("flash");
  void target.offsetWidth; // restart the highlight when the same marker is clicked twice
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1600);
}

// Split the text on [n] markers and build nodes: markers with a matching source become links,
// everything else (including unknown markers) stays a plain text node.
function textWithCitations(text, count) {
  const box = el("div", { class: "answer" });
  for (const part of text.split(/(\[\d+\])/)) {
    const m = part.match(/^\[(\d+)\]$/);
    const n = m ? Number(m[1]) : 0;
    if (!m || n < 1 || n > count) { if (part) box.append(part); continue; }
    const a = el("a", { href: `#src-${n}`, title: t("ui.ticket.source_link", { n }) }, part);
    a.addEventListener("click", (ev) => { ev.preventDefault(); jumpTo(n); });
    box.append(a);
  }
  return box;
}

function sourceList(citations, numbered) {
  return el("ol", { class: "sources" }, ...citations.map((c, i) => el("li", { class: "source", id: numbered ? `src-${i + 1}` : null },
    el("div", { class: "meta" },
      numbered ? el("span", { class: "num" }, `[${i + 1}]`) : null,
      el("span", { class: "file" }, c.file),
      c.page != null ? el("span", {}, t("ui.ticket.page", { n: c.page })) : null),
    el("p", { class: "fragment" }, c.fragment),
  )));
}

function escalationPlate(draft) {
  const plate = el("div", { class: "plate escalated", role: "status" },
    el("h3", {}, `⚠ ${t("ui.ticket.escalated_title")}`),
    el("p", { class: "reason" }, reasonText(draft.escalation_reason)),
    el("p", { class: "todo" }, t("ui.ticket.escalated_hint")),
  );
  // The server's own note (Russian, may carry the model's explanation or the LLM error code).
  const note = el("details", { class: "note" }, el("summary", {}, t("ui.ticket.system_note")), el("pre", {}, draft.text));
  plate.append(note);
  return plate;
}

function generateBlock(ticket) {
  const busy = drafting.has(ticket.id);
  const button = el("button", { class: `primary${busy ? " busy" : ""}`, type: "button", disabled: busy });
  if (busy) button.append(el("span", { class: "spinner", "aria-hidden": "true" }), t("ui.ticket.generating"));
  else button.append(t(ticket.drafts.length ? "ui.ticket.regenerate" : "ui.ticket.generate"));
  button.addEventListener("click", () => generateDraft(ticket.id));
  return el("div", { class: "generate" }, button,
    busy ? el("p", { class: "hint", "aria-live": "polite" }, t("ui.ticket.generating_hint")) : null);
}

function actionsBlock(ticket) {
  const reply = replyDraft(ticket);
  const copy = el("button", { class: "ghost", type: "button", disabled: !reply, title: reply ? null : t("ui.ticket.copy_disabled") },
    t("ui.ticket.copy"));
  copy.addEventListener("click", () => copyReply(ticket.id, reply.text));
  const box = el("div", { class: "actions" }, copy);

  if (ticket.status === "resolved") return box;
  const canResolve = ticket.status === "drafted" && !drafting.has(ticket.id);
  if (confirmingResolve && canResolve) {
    const yes = el("button", { class: "primary", type: "button" }, t("ui.ticket.resolve_confirm"));
    const no = el("button", { class: "ghost", type: "button" }, t("ui.ticket.resolve_cancel"));
    yes.addEventListener("click", () => resolveTicket(ticket.id, yes));
    no.addEventListener("click", () => { confirmingResolve = false; renderTicket(); });
    box.append(el("p", { class: "hint" }, t("ui.ticket.resolve_question")), yes, no);
    return box;
  }
  const resolve = el("button", { class: "ghost", type: "button", disabled: !canResolve }, t("ui.ticket.resolve"));
  resolve.addEventListener("click", () => { confirmingResolve = true; renderTicket(); });
  box.append(resolve);
  if (!canResolve && !drafting.has(ticket.id)) box.append(el("p", { class: "hint" }, t("ui.ticket.resolve_disabled")));
  return box;
}

function historyBlock(ticket) {
  const drafts = ticket.drafts;
  if (drafts.length < 2) return null;
  const items = [...drafts].reverse().map((d, i) => el("li", {}, el("details", {},
    el("summary", {},
      el("span", { class: "when" }, formatDate(d.created_at)),
      d.confidence === "answered"
        ? el("span", { class: "badge s-drafted" }, t("ui.ticket.history_answered"))
        : el("span", { class: "badge s-escalated" }, t("ui.ticket.history_escalated")),
      i === 0 ? el("span", { class: "why" }, `(${t("ui.ticket.latest")})`) : null,
      d.escalation_reason ? el("span", { class: "why" }, reasonText(d.escalation_reason)) : null),
    el("pre", {}, d.text),
  )));
  return el("details", { class: "history" }, el("summary", {}, t("ui.ticket.history", { n: drafts.length })), el("ol", {}, ...items));
}

function draftSection(ticket) {
  const draft = latestDraft(ticket);
  const parts = [];
  if (ticket.status !== "resolved") parts.push(generateBlock(ticket));
  if (cardNotice && cardNotice.id === ticket.id) parts.push(message(cardNotice.kind, cardNotice.text()));

  if (ticket.status === "resolved") {
    parts.push(el("div", { class: "plate resolved", role: "status" },
      el("h3", {}, `✓ ${t("ui.ticket.resolved_title")}`), el("p", {}, t("ui.ticket.resolved_hint"))));
  }
  if (!draft) {
    parts.push(el("p", { class: "hint" }, t("ui.ticket.no_drafts")));
  } else if (draft.confidence === "answered") {
    parts.push(
      el("div", { class: "label" }, t("ui.ticket.draft")),
      textWithCitations(draft.text, draft.citations.length),
      el("p", { class: "hint" }, t("ui.ticket.draft_hint")),
    );
    if (draft.citations.length) parts.push(el("div", { class: "label" }, t("ui.ticket.sources")), sourceList(draft.citations, true));
  } else {
    parts.push(escalationPlate(draft));
    if (draft.citations.length) {
      parts.push(el("div", { class: "label" }, t("ui.ticket.found")),
        el("p", { class: "hint" }, t("ui.ticket.found_hint")), sourceList(draft.citations, false));
    }
  }
  if (draft) parts.push(actionsBlock(ticket));
  const history = historyBlock(ticket);
  if (history) parts.push(history);
  return parts;
}

// --- ticket card: actions ---------------------------------------------------------------
async function generateDraft(id) {
  if (drafting.has(id)) return;
  drafting.add(id);
  confirmingResolve = false;
  if (cardNotice && cardNotice.id === id) cardNotice = null;
  renderTicket();
  try {
    const { ticket, draft } = await api(`/tickets/${id}/draft`, { method: "POST" });
    if (ticketDetail && ticketDetail.id === id) {
      ticketDetail = { ...ticketDetail, ...ticket, drafts: [...ticketDetail.drafts, draft] };
    }
    updateTicketInList(ticket);
  } catch (e) {
    cardNotice = { id, kind: "err", text: () => errorText(e) };
  } finally {
    drafting.delete(id);
    renderTicket();
  }
}

// The Clipboard API needs a secure context (127.0.0.1 is one, a LAN address is not) and can hang
// on a permission prompt, so it gets a deadline and falls back to a hidden textarea.
function copyWithTextarea(text) {
  const area = el("textarea", { readonly: true, "aria-hidden": "true", style: "position:fixed;top:0;left:0;opacity:0" });
  area.value = text;
  document.body.append(area);
  area.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { ok = false; }
  area.remove();
  return ok;
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    const deadline = new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), 1500));
    try {
      await Promise.race([navigator.clipboard.writeText(text), deadline]);
      return true;
    } catch { /* fall through */ }
  }
  return copyWithTextarea(text);
}

async function copyReply(id, text) {
  const ok = await copyText(text);
  cardNotice = ok
    ? { id, kind: "ok", text: () => t("ui.ticket.copied") }
    : { id, kind: "warn", text: () => t("ui.ticket.copy_failed") };
  renderTicket();
}

async function resolveTicket(id, button) {
  button.disabled = true;
  button.textContent = t("ui.ticket.resolving");
  try {
    const ticket = await api(`/tickets/${id}/resolve`, { method: "POST" });
    if (ticketDetail && ticketDetail.id === id) ticketDetail = ticket;
    updateTicketInList(ticket);
    cardNotice = { id, kind: "ok", text: () => t("ui.ticket.resolved_ok", { id }) };
  } catch (e) {
    cardNotice = { id, kind: "err", text: () => errorText(e) };
  } finally {
    confirmingResolve = false;
    renderTicket();
  }
}

function selectedFromHash() {
  const m = location.hash.match(/^#ticket-(\d+)$/);
  return m ? Number(m[1]) : null;
}

// --- startup --------------------------------------------------------------------
function reloadAll() {
  refreshStatus();
  loadDocuments();
  loadTickets();
  if (selectedId) selectTicket(selectedId);
}

window.SC.onLanguageChange(() => {
  renderStatus();
  renderDocuments();
  renderUploadLog();
  renderFilters();
  renderTickets();
  renderFormMessages();
  renderTicket();
});

(async () => {
  await window.SC.loadTranslations();
  window.SC.mountControls($("controls"));
  selectedId = selectedFromHash();
  reloadAll();
})();
