const $ = (id) => document.getElementById(id);
const state = () => ({
  causal: $("causal").checked,
  window: $("window").checked,
  windowSize: Number($("window-size").value),
  gqa: $("gqa").checked,
  softcap: $("softcap").checked,
  alibi: $("alibi").checked,
  headDim: Number($("head-dim").value),
  arch: $("arch").value,
});

function escapeHtml(value) {
  return value.replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

function highlight(code) {
  return escapeHtml(code)
    .replace(/\b(from|import|def|return|const|float|int|if|for|continue|extern|true|false|null)\b/g, '<span class="kw">$1</span>')
    .replace(/\b(attention|compile|tanhf|expf|classify_kv_tile|warp_sum)\b/g, '<span class="fn">$1</span>')
    .replace(/(&quot;.*?&quot;)/g, '<span class="str">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?f?)\b/g, '<span class="num">$1</span>')
    .replace(/(\/\/.*|#.*)/g, '<span class="cm">$1</span>');
}

function dsl(s) {
  const calls = [];
  if (s.causal) calls.push(".causal()");
  if (s.window) calls.push(`.sliding_window(${s.windowSize})`);
  if (s.gqa) calls.push(".gqa(kv_heads=2)");
  if (s.softcap) calls.push(".softcap(50.0)");
  if (s.alibi) calls.push(".alibi(slopes)");
  return `from attnc import attention, compile

spec = (attention(head_dim=${s.headDim}, dtype="fp16")
        ${calls.join("\n        ") || "# vanilla attention"})

kernel = compile(spec, arch="${s.arch}")
out = kernel(q, k, v)`;
}

function ir(s) {
  const mask = [s.causal && "q_idx >= kv_idx", s.window && `q_idx - kv_idx < ${s.windowSize}`].filter(Boolean);
  let score = "score";
  if (s.alibi) score = "sub(score, mul(lookup(slopes, h), sub(q_idx, kv_idx)))";
  if (s.softcap) score = `mul(50.0, tanh(div(${score}, 50.0)))`;
  return JSON.stringify({
    layout: {head_dim:s.headDim, dtype:"fp16", kv_heads:s.gqa ? 2 : null, kv_layout:"contiguous"},
    mask: mask.length ? mask.join(" AND ") : true,
    score,
    bounds: (s.causal || s.window) ? {causal:s.causal, window:s.window ? s.windowSize : null} : null,
    passes: ["constant_fold", "simplify_bool", "infer_tile_bounds"]
  }, null, 2);
}

function cuda(s) {
  const pred = [s.causal && "q_idx >= kv_idx", s.window && `q_idx - kv_idx < ${s.windowSize}`].filter(Boolean).join(" && ") || "true";
  let mod = "score";
  if (s.alibi) mod = "score - slopes[h] * (q_idx - kv_idx)";
  if (s.softcap) mod = `50.0f * tanhf((${mod}) / 50.0f)`;
  return `// generated inner loop · ${s.arch} · HEAD_DIM=${s.headDim}
for (int k0 = 0; k0 < NK; k0 += KV_TILE) {
  int tile = classify_kv_tile(q_idx, k0, k1);
  if (tile == FULLY_MASKED) continue;

  for (int kv_idx = k0; kv_idx < k1; ++kv_idx) {
    if (tile == PARTIAL && !(${pred})) continue;

    float score = warp_sum(dot(q, k)) * scale;
    score = ${mod};

    float next_m = fmaxf(m, score);
    float alpha = expf(m - next_m);
    float beta  = expf(score - next_m);
    acc = acc * alpha + beta * v;
    l = l * alpha + beta;
    m = next_m;
  }
}`;
}

function renderTiles(s) {
  const grid = $("tile-grid");
  grid.innerHTML = "";
  for (let q = 0; q < 12; q++) {
    for (let k = 0; k < 12; k++) {
      let kind = "full";
      if (s.causal && k > q) kind = "masked";
      if (s.window && q - k > 4) kind = "masked";
      if (kind === "full" && ((s.causal && k === q) || (s.window && q - k === 4))) kind = "partial";
      const cell = document.createElement("i");
      cell.className = `tile ${kind}`;
      cell.title = `q tile ${q}, KV tile ${k}: ${kind}`;
      grid.appendChild(cell);
    }
  }
}

function render() {
  const s = state();
  $("window-value").value = s.windowSize;
  $("window-wrap").style.opacity = s.window ? "1" : ".32";
  $("dsl").innerHTML = highlight(dsl(s));
  $("ir").innerHTML = highlight(ir(s));
  $("cuda").innerHTML = highlight(cuda(s));
  renderTiles(s);
  $("tile-caption").textContent = (s.causal || s.window)
    ? "Recognized bounds become a closed-form key interval. Fully masked tiles issue no memory traffic."
    : "An unrestricted mask sends every tile through the fast, unpredicated path.";
}

document.querySelectorAll("input, select").forEach((el) => el.addEventListener("input", render));
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab,.code").forEach((el) => el.classList.remove("active"));
  tab.classList.add("active");
  $(tab.dataset.tab).classList.add("active");
}));
$("compile-button").addEventListener("click", () => {
  const button = $("compile-button");
  button.classList.add("compiling");
  $("status-text").textContent = "lowering IR · selecting tiles…";
  setTimeout(() => {
    button.classList.remove("compiling");
    $("status-text").textContent = `compiled · ${state().arch} · cache stored`;
    render();
  }, 650);
});
$("copy").addEventListener("click", async () => {
  const active = document.querySelector(".code.active");
  await navigator.clipboard.writeText(active.innerText);
  $("copy").textContent = "COPIED";
  setTimeout(() => $("copy").textContent = "COPY", 900);
});
render();
