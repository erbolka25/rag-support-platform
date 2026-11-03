// Theme, translations, DOM and API helpers shared by the operator UI.
// Loaded in <head> without defer, so the saved theme applies before the first paint.
// Everything that comes from users or the server is inserted as text (el() / textContent), never as HTML.
(() => {
  const DEFAULT_LANG = "ru";
  const THEMES = ["auto", "light", "dark"];

  const store = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage blocked */ } },
  };

  // --- theme ------------------------------------------------------------------
  function currentTheme() {
    const saved = store.get("theme");
    return THEMES.includes(saved) ? saved : "auto";
  }

  function applyTheme(theme) {
    if (theme === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
  }

  applyTheme(currentTheme());

  // --- translations -----------------------------------------------------------
  let catalog = {};
  let lang = DEFAULT_LANG;
  const listeners = [];

  function lookup(tree, key) {
    return key.split(".").reduce((node, part) => (node && typeof node === "object" ? node[part] : undefined), tree);
  }

  function has(key) {
    return typeof lookup(catalog[lang], key) === "string" || typeof lookup(catalog[DEFAULT_LANG], key) === "string";
  }

  function t(key, params = {}) {
    let text = lookup(catalog[lang], key);
    if (typeof text !== "string") text = lookup(catalog[DEFAULT_LANG], key);
    if (typeof text !== "string") return key;
    return text.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
  }

  function applyTranslations(root = document) {
    root.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
    for (const attr of ["placeholder", "title", "aria-label"]) {
      root.querySelectorAll(`[data-i18n-${attr}]`).forEach((node) => {
        node.setAttribute(attr, t(node.getAttribute(`data-i18n-${attr}`)));
      });
    }
    document.documentElement.lang = lang;
  }

  async function loadTranslations() {
    try {
      catalog = await (await fetch("/static/i18n.json")).json();
    } catch {
      catalog = {};
    }
    const wanted = new URLSearchParams(location.search).get("lang") || store.get("lang");
    lang = wanted && wanted in catalog ? wanted : DEFAULT_LANG;
    applyTranslations();
    document.documentElement.classList.remove("i18n-pending");
  }

  function setLanguage(code) {
    if (!(code in catalog)) return;
    lang = code;
    store.set("lang", code);
    const url = new URL(location.href);
    if (url.searchParams.has("lang")) {
      url.searchParams.delete("lang");
      history.replaceState(null, "", url);
    }
    applyTranslations();
    listeners.forEach((fn) => fn(code));
  }

  function formatDate(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleString(lang === "ru" ? "ru-RU" : "en-GB", { dateStyle: "short", timeStyle: "short" });
  }

  // --- DOM and API helpers ------------------------------------------------------
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v == null) continue;
      if (k === "class") node.className = v;
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children) if (c != null) node.append(c);
    return node;
  }

  // Drafting waits for the LLM, which can take a while; past this the request counts as failed.
  const API_TIMEOUT_MS = 90000;
  // Statuses a proxy returns when the app itself is down.
  const UNREACHABLE_STATUSES = [502, 503, 504];

  // The API answers errors as {"detail": {"code", "message"}}, or {"detail": [...]} for validation.
  // The message shown is always ours (by code, then by status): the server's text is English-only.
  class ApiError extends Error {
    constructor(status, code, detail = null) {
      // A proxy in front of the app rejects big uploads with a bare 413, without our JSON body.
      if (!code && status === 413) code = "file_too_large";
      let message;
      if (code && has(`ui.errors.${code}`)) message = t(`ui.errors.${code}`);
      else if (status === 0) message = t("ui.network.unreachable");
      else if (UNREACHABLE_STATUSES.includes(status)) message = t("ui.network.unreachable");
      else message = t("ui.errors.status", { status });
      super(message);
      this.status = status;
      this.code = code;
      this.detail = detail; // the validation error list for a 422, so forms can point at a field
    }
  }

  function networkError(e) {
    return new ApiError(0, e && e.name === "TimeoutError" ? "timeout" : "unreachable");
  }

  async function api(path, options = {}) {
    const signal = typeof AbortSignal.timeout === "function" ? AbortSignal.timeout(API_TIMEOUT_MS) : undefined;
    let resp;
    try {
      resp = await fetch(path, { signal, ...options });
    } catch (e) {
      throw networkError(e);
    }
    let body = null;
    try {
      body = await resp.json();
    } catch (e) {
      // The connection can drop or time out while the body is still arriving.
      if (e && (e.name === "TimeoutError" || e.name === "AbortError")) throw networkError(e);
    }
    if (!resp.ok) {
      const detail = body && body.detail;
      let code = null;
      if (detail && typeof detail.code === "string") code = detail.code;
      else if (Array.isArray(detail)) code = "validation";
      throw new ApiError(resp.status, code, Array.isArray(detail) ? detail : null);
    }
    if (body === null || typeof body !== "object") throw new ApiError(resp.status, "bad_response");
    return body;
  }

  // Language and theme selects.
  function mountControls(container) {
    const langSelect = el("select", { "data-i18n-aria-label": "ui.controls.language" });
    for (const code of Object.keys(catalog)) {
      const name = (catalog[code].meta && catalog[code].meta.name) || code;
      langSelect.append(el("option", { value: code }, name));
    }
    langSelect.value = lang;
    langSelect.addEventListener("change", () => setLanguage(langSelect.value));

    const themeSelect = el("select", { "data-i18n-aria-label": "ui.controls.theme" });
    for (const theme of THEMES) {
      themeSelect.append(el("option", { value: theme, "data-i18n": `ui.controls.theme_${theme}` }));
    }
    themeSelect.value = currentTheme();
    themeSelect.addEventListener("change", () => {
      store.set("theme", themeSelect.value);
      applyTheme(themeSelect.value);
    });

    container.replaceChildren(langSelect, themeSelect);
    applyTranslations(container);
  }

  window.SC = {
    t,
    has,
    el,
    api,
    formatDate,
    applyTranslations,
    loadTranslations,
    mountControls,
    store,
    lang: () => lang,
    onLanguageChange: (fn) => listeners.push(fn),
  };
})();
