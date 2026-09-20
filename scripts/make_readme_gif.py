"""Generate the exact, lightweight README demo animation (no external assets)."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "attnc-demo.gif"
W, H = 1000, 560


def font(size: int, mono: bool = False, bold: bool = False):
    if mono:
        name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    else:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


F9, F10, F11, F12, F14, F16, F22 = (font(s) for s in (9, 10, 11, 12, 14, 16, 22))
M9, M10, M11, M12, M14 = (font(s, True) for s in (9, 10, 11, 12, 14))
M10B, M13B = font(10, True, True), font(13, True, True)

BG, PANEL, PANEL2, LINE = "#080a13", "#101522", "#0b0f1a", "#283047"
WHITE, TEXT, MUTED = "#f7f8fb", "#d9deeb", "#78829b"
CYAN, PURPLE, GREEN, ORANGE = "#42e8e0", "#a78bfa", "#70f0a9", "#ffb86b"


def rr(d, box, radius=10, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius, fill=fill, outline=outline, width=width)


def toggle(d, y, label, sub, on):
    d.text((38, y), label, font=F11, fill=WHITE)
    d.text((38, y + 16), sub, font=F9, fill=MUTED)
    rr(d, (204, y + 5, 235, y + 22), 9, fill="#16353a" if on else "#262b3b", outline="#367e7d" if on else None)
    x = 221 if on else 208
    d.ellipse((x, y + 9, x + 9, y + 18), fill=CYAN if on else "#6a7288")


def line(d, x, y, parts):
    for text, color in parts:
        d.text((x, y), text, font=M10, fill=color)
        x += d.textlength(text, font=M10)


def frame(step: int):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    # glow fields
    for r in range(170, 10, -12):
        alpha = int((170 - r) * .04)
        c = (8 + alpha, 18 + alpha, 29 + alpha)
        d.ellipse((-60 - r//2, -100 - r//2, 160 + r, 120 + r), fill=c)
    d.rectangle((0, 0, W, 56), fill="#0b0e18")
    d.line((0, 55, W, 55), fill=LINE)
    d.text((26, 18), "a/", font=M14, fill=CYAN)
    d.text((47, 18), "attnc", font=M14, fill=WHITE)
    d.text((797, 20), "ATTENTION COMPILER PLAYGROUND", font=M9, fill=MUTED)
    # pipeline header
    stages = [("01", "CAPTURE"), ("02", "OPTIMIZE"), ("03", "CLASSIFY"), ("04", "EMIT")]
    for i, (num, name) in enumerate(stages):
        x = 286 + i * 165
        active = i <= min(step // 3, 3)
        d.ellipse((x, 76, x + 8, 84), fill=CYAN if active else "#353c50")
        d.text((x + 15, 72), f"{num} / {name}", font=M9, fill=CYAN if active else MUTED)
        if i < 3: d.line((x + 110, 80, x + 153, 80), fill="#354057")
    # panels
    rr(d, (22, 108, 258, 507), 11, PANEL, LINE)
    rr(d, (274, 108, 708, 507), 11, PANEL2, LINE)
    rr(d, (724, 108, 978, 507), 11, PANEL, LINE)
    d.text((37, 127), "VARIANT SPEC", font=M10B, fill=MUTED)
    d.line((37, 148, 242, 148), fill=LINE)
    states = {
        "causal": True,
        "window": step >= 2,
        "gqa": step >= 5,
        "softcap": step >= 8,
        "alibi": step >= 11,
    }
    toggle(d, 166, "Causal", "mask future keys", states["causal"])
    toggle(d, 216, "Sliding window", "bounded KV history", states["window"])
    toggle(d, 266, "GQA", "8 query → 2 KV heads", states["gqa"])
    toggle(d, 316, "Logit softcap", "cap · tanh(score/cap)", states["softcap"])
    toggle(d, 366, "ALiBi", "head-specific bias", states["alibi"])
    rr(d, (37, 427, 242, 464), 6, CYAN)
    d.text((65, 440), "▶  COMPILE VARIANT", font=M10B, fill="#071012")
    d.ellipse((71, 482, 77, 488), fill=GREEN)
    d.text((84, 479), "cache stored · sm_121", font=M9, fill=MUTED)

    tab_idx = min(step // 5, 2)
    tabs = ["PYTHON DSL", "OPTIMIZED IR", "CUDA"]
    for i, label in enumerate(tabs):
        x = 293 + i * 128
        d.text((x, 128), label, font=M9, fill=CYAN if i == tab_idx else MUTED)
        if i == tab_idx: d.line((x, 148, x + 80, 148), fill=CYAN, width=2)
    d.line((288, 154, 694, 154), fill=LINE)
    y = 178
    if tab_idx == 0:
        calls = [".causal()"]
        if states["window"]: calls.append(".sliding_window(4096)")
        if states["gqa"]: calls.append(".gqa(kv_heads=2)")
        if states["softcap"]: calls.append(".softcap(50.0)")
        if states["alibi"]: calls.append(".alibi(slopes)")
        code = [
            [("from ", PURPLE), ("attnc ", TEXT), ("import ", PURPLE), ("attention, compile", CYAN)],
            [], [("spec = (", TEXT), ("attention", CYAN), ("(head_dim=", TEXT), ("128", ORANGE), (")", TEXT)],
        ] + [[("        ", TEXT), (c, CYAN)] for c in calls] + [
            [("kernel = ", TEXT), ("compile", CYAN), ("(spec, arch=", TEXT), ('"sm_121"', GREEN), (")", TEXT)],
            [("out = kernel(q, k, v)", TEXT)],
        ]
        for row in code:
            line(d, 297, y, row); y += 25
    elif tab_idx == 1:
        fields = [
            ("{", TEXT), ('  "layout"', PURPLE), (": {", TEXT),
            ('    "head_dim"', CYAN), (": 128,", TEXT),
            ('    "dtype"', CYAN), (': "fp16",', GREEN),
            ('    "kv_heads"', CYAN), (": 2", ORANGE), ("  },", TEXT),
            ('  "mask"', PURPLE), (': "causal AND window(4096)",', GREEN),
            ('  "score"', PURPLE), (': "softcap(alibi(score))",', GREEN),
            ('  "tile_states"', PURPLE), (': ["SKIP", "FAST", "PRED"]', GREEN), ("}", TEXT),
        ]
        for text, color in fields:
            d.text((297, y), text, font=M10, fill=color); y += 22
    else:
        rows = [
            ("for (int k0 = 0; k0 < NK; k0 += 64) {", PURPLE),
            ("  int tile = classify_kv_tile(q, k0, k1);", TEXT),
            ("  if (tile == FULLY_MASKED) continue;", ORANGE),
            ("", TEXT),
            ("  float score = warp_sum(dot(q, k)) * scale;", TEXT),
            ("  score = 50.f * tanhf(score / 50.f);", CYAN),
            ("  float next_m = fmaxf(m, score);", TEXT),
            ("  acc = acc * alpha + beta * v;", GREEN),
            ("  l = l * alpha + beta;", GREEN),
            ("}", PURPLE),
        ]
        for text, color in rows:
            d.text((297, y), text, font=M10, fill=color); y += 24

    d.text((740, 127), "STATIC TILE PLAN", font=M10B, fill=MUTED)
    d.line((740, 148, 962, 148), fill=LINE)
    grid_x, grid_y, cell, gap = 740, 174, 15, 3
    for q in range(12):
        for k in range(12):
            kind = "full"
            if k > q: kind = "masked"
            if states["window"] and q - k > 4: kind = "masked"
            if kind == "full" and (k == q or (states["window"] and q - k == 4)): kind = "partial"
            color = {"full": CYAN, "partial": PURPLE, "masked": "#1d2333"}[kind]
            x, yy = grid_x + k * (cell + gap), grid_y + q * (cell + gap)
            rr(d, (x, yy, x + cell, yy + cell), 2, color)
    d.rectangle((740, 405, 748, 413), fill=CYAN); d.text((754, 402), "fast", font=M9, fill=MUTED)
    d.rectangle((800, 405, 808, 413), fill=PURPLE); d.text((814, 402), "pred", font=M9, fill=MUTED)
    d.rectangle((866, 405, 874, 413), fill="#283047"); d.text((880, 402), "skipped", font=M9, fill=MUTED)
    d.text((740, 437), "closed-form bounds", font=M10, fill=WHITE)
    d.text((740, 455), "remove predicates and", font=M9, fill=MUTED)
    d.text((740, 470), "skip memory traffic", font=M9, fill=MUTED)
    # progress and footer
    d.line((22, 529, 978, 529), fill=LINE)
    progress = 22 + int(956 * (step + 1) / 15)
    d.line((22, 529, progress, 529), fill=CYAN, width=2)
    d.text((22, 540), "COMPOSE → LOWER → OPTIMIZE → EMIT", font=M9, fill=MUTED)
    d.text((789, 540), "160 / 160 TESTS PASSING", font=M9, fill=GREEN)
    return im


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame(i) for i in range(15)]
    palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    palette[0].save(OUT, save_all=True, append_images=palette[1:], duration=[520] * 14 + [1200], loop=0, optimize=True, disposal=2)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
