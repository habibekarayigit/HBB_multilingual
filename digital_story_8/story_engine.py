#!/usr/bin/env python3
"""Sahne 3-5 icin ortak montaj motoru (yardimci fonksiyonlar build_scene1.py'den gelir).

Her sahne bir PLAN listesi verir. Her cekim bir sozluktur:
  img    : gorsel dosyasi (veya aciklama)
  c0,c1  : baslangic/bitis kadraji (0-1 oranlari)  -> yavas kamera hareketi
  lid    : replik kimligi (None = repliksiz)
  rule   : sure (sn) | "line" (tum replik) | ("f", oran) (repligin bir bolumu) | "tail" (repligin kalani)
  anim   : None (sabit kare) | "card" | "wide" | FACES anahtari (agiz+goz animasyonu) | fonksiyon(tt, seg, ctx)
  overlay: fonksiyon(kare, tt, seg, ctx) -> kare   (kadrajdan sonra ustune cizilir)
  note   : cekim listesi aciklamasi
"""
import os, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import build_scene1 as b
from build_scene1 import W, H, FPS, SR, GREEN, FF, font, ease, fade_text

FULL = (0, 0, 1, 1)
GAP = 0.40


def auto_box(a, bimg, pad=28):
    """Iki kare arasindaki farktan yuz kutusunu bulur."""
    A = np.asarray(Image.open(a).convert("RGB"), float); B = np.asarray(Image.open(bimg).convert("RGB").resize((A.shape[1], A.shape[0])), float)
    m = np.asarray(Image.fromarray(((np.abs(A - B).mean(2) > 45) * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))) > 0
    ys, xs = np.where(m)
    x0, x1 = np.percentile(xs, [1, 99]); y0, y1 = np.percentile(ys, [1, 99])
    return (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))


def split_clip(path, n_chars):
    """Ayni sesle tek kayitta okunan iki repligi, beklenen yere en yakin uzun sessizlikten boler."""
    a = b.load_wav(path); hop = SR // 100
    env = np.array([np.abs(a[i:i + hop]).max() for i in range(0, len(a), hop)])
    sil = env < 0.02; gaps = []; i = 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]: j += 1
            if (j - i) > 15: gaps.append(((i + j) / 2 * hop, (j - i) / 100))
            i = j
        else: i += 1
    expect = len(a) * n_chars[0] / sum(n_chars)
    mid, glen = min(gaps, key=lambda g: abs(g[0] - expect) - g[1] * SR * 0.5)
    assert abs(mid - expect) < 0.15 * len(a) and glen > 0.35, (path, gaps, expect / SR)
    cut = int(mid); pad = int(0.12 * SR)
    return a[:cut - int(glen * SR / 2) + pad], a[cut + int(glen * SR / 2) - int(0.05 * SR):]


def inset(img_path, f0, f1, corner="tl", size=400):
    """Konu gorselini kose penceresi olarak gosterir (replik suresinin f0-f1 araliginda)."""
    cache = {}
    def draw(frame, tt, seg, ctx):
        if seg["lid"] is None: return frame
        st = ctx["placed"][seg["lid"]]; L = len(ctx["audio"][seg["lid"]]) / SR
        t = seg["start"] + tt - st; t0, t1 = f0 * L, f1 * L
        if not (t0 <= t <= t1): return frame
        a = min(ease((t - t0) / 0.3), ease((t1 - t) / 0.25))
        if a <= 0: return frame
        if "im" not in cache:
            h = size * 9 // 16
            pic = Image.open(img_path).convert("RGB").resize((size, h), Image.LANCZOS)
            card = Image.new("RGBA", (size + 16, h + 16), (0, 0, 0, 0))
            ImageDraw.Draw(card).rounded_rectangle((0, 0, size + 15, h + 15), radius=18, fill=(255, 255, 255, 255))
            mk = Image.new("L", (size, h), 0); ImageDraw.Draw(mk).rounded_rectangle((0, 0, size - 1, h - 1), radius=12, fill=255)
            card.paste(pic, (8, 8), mk); cache["im"] = card
        card = cache["im"]; s = 0.6 + 0.4 * a
        c = card.resize((max(2, int(card.width * s)), max(2, int(card.height * s))), Image.BICUBIC)
        c.putalpha(c.getchannel("A").point(lambda v: int(v * a)))
        x = 36 if corner[1] == "l" else W - 36 - card.width
        y = 36 if corner[0] == "t" else H - 36 - card.height
        x += (card.width - c.width) // 2; y += (card.height - c.height) // 2
        out = frame.convert("RGBA")
        sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((x + 6, y + 8, x + c.width + 6, y + c.height + 8), radius=18, fill=(0, 0, 0, int(90 * a)))
        out.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6))); out.alpha_composite(c, (x, y))
        return out.convert("RGB")
    return draw


def render(num, title, LINES, audio, PLAN, sfx=(), md_extra="", rules_md=""):
    """LINES: lid -> (konusan, metin, ses dosyasi adi); audio: lid -> ornek dizisi."""
    segs, t, placed = [], 0.0, {}
    for i, s in enumerate(PLAN):
        s = dict(s); lid, rule = s.get("lid"), s["rule"]
        if lid is None:
            dur = rule
        else:
            L = len(audio[lid]) / SR
            if rule == "tail":
                dur = placed[lid] + L + GAP - t
            else:
                placed.setdefault(lid, t)
                dur = L + GAP if rule == "line" else rule[1] * L
        s.update(start=t, dur=dur, idx=i); s.setdefault("c0", FULL); s.setdefault("c1", s["c0"]); s.setdefault("anim", None)
        segs.append(s); t += dur
    total = t
    voice = np.zeros(int((total + 0.5) * SR), dtype=np.float32)
    for lid, st in placed.items():
        a = audio[lid]; i = int(st * SR); voice[i:i + len(a)] += a
    mix = voice.copy()
    for item in sfx:
        st, a = item(total) if callable(item) else item
        i = int(st * SR); mix[i:i + len(a)] += a[: len(mix) - i]
    mix = np.clip(mix, -0.98, 0.98)
    tmpwav = f"/tmp/scene{num}_mix.wav"
    with wave.open(tmpwav, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
    tracks = {k: b.mouth_track(a) for k, a in audio.items()}
    ctx = dict(placed=placed, audio=audio, voice=voice, total=total)
    cache = {}

    def source(seg, tt):
        an = seg["anim"]
        if callable(an): return an(tt, seg, ctx)
        if an == "card": return b.card_frame(tt, seg["dur"], num, title)
        if an == "wide" or an in b.FACES:
            fs = b.wide_set() if an == "wide" else b.face_set(an); closed = 1
            if seg["lid"]:
                i = int(round((seg["start"] + tt - placed[seg["lid"]]) * FPS)); tr = tracks[seg["lid"]]
                if 0 <= i < len(tr) and tr[i]: closed = 0
            return fs[(closed, 1 if b.blink_at(seg["start"] + tt, seg["idx"]) else 0)]
        if seg["img"] not in cache: cache[seg["img"]] = Image.open(seg["img"]).convert("RGB")
        return cache[seg["img"]]

    def frame_of(seg, tt):
        tt = min(tt, seg["dur"] - 1e-3)
        im = source(seg, tt); p = ease(tt / seg["dur"])
        c = [a + (bb - a) * p for a, bb in zip(seg["c0"], seg["c1"])]
        if not (c == [0, 0, 1, 1] and im.size == (W, H)):
            im = im.resize((W, H), Image.BICUBIC, box=(c[0] * im.width, c[1] * im.height, c[2] * im.width, c[3] * im.height))
        if seg.get("overlay"): im = seg["overlay"](im, tt, seg, ctx)
        return im

    out = f"DigitalStory8_Scene{num}.mp4"
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-i", tmpwav, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                          "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    XF = 0.3; black = Image.new("RGB", (W, H))
    for i in range(int(total * FPS)):
        tt = i / FPS
        k = max(j for j, s in enumerate(segs) if s["start"] <= tt + 1e-9)
        s = segs[k]; local = tt - s["start"]
        fr = frame_of(s, local)
        if k > 0 and local < XF and (segs[k - 1]["img"] != s["img"] or segs[k - 1]["c1"] != s["c0"]):
            fr = Image.blend(frame_of(segs[k - 1], segs[k - 1]["dur"]), fr, local / XF)
        if tt < 0.4: fr = Image.blend(black, fr, tt / 0.4)
        if total - tt < 0.8: fr = Image.blend(black, fr, max(0, (total - tt) / 0.8))
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()

    def tc(x): return f"{int(x // 60):02d}:{x % 60:04.1f}"
    rows = ["| # | Zaman | Görsel dosyası | Ekranda ne var / kamera | Konuşan | Replik (aynen okunur) | Ses |", "|---|---|---|---|---|---|---|"]
    said = set()
    for i, s in enumerate(segs, 1):
        who = line = ""; snd = "—"
        if s.get("lid"):
            who, text, snd = LINES[s["lid"]]
            line = "(önceki replik sürer)" if s["lid"] in said else text; said.add(s["lid"])
        rows.append(f"| {i} | {tc(s['start'])}–{tc(s['start'] + s['dur'])} | `{s['img']}` | {s['note']} | {who} | {line} | {snd} |")
    md = f"""# Digital Story 8 – Scene {num} "{title}" – Çekim Listesi

**Toplam süre:** {tc(total)} (sahne geçiş karesi dâhil)
**Format:** 1280×720, 25 fps, MP4 (H.264 + AAC)

Karakterler önceki sahnelerle aynıdır (`characters/character_sheet.png`). Kıyafetlerin görünümü korunur.
{md_extra}
## Çekim listesi

{chr(10).join(rows)}

## Talimat belgesindeki kurallara uyum

{rules_md}"""
    open(f"CEKIM_LISTESI_Scene{num}.md", "w", encoding="utf-8").write(md)
    print("OK", out, "sure:", tc(total))
    return segs


def align_split(path, lines, extra_words=None):
    """Tek kayitta okunan birden cok repligi, kelime hizalama (pocketsphinx) ile replik sinirlarindan boler."""
    import re
    from pocketsphinx import Decoder
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "/tmp/_al.wav"], check=True)
    with wave.open("/tmp/_al.wav") as w: data = w.readframes(w.getnframes())
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", path, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", "/tmp/_al2.wav"], check=True)
    with wave.open("/tmp/_al2.wav") as w: a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    rms = np.sqrt(np.mean(a[np.abs(a) > 0.01] ** 2)); a = a * (0.1 / rms)
    pk = np.abs(a).max(); a = a * 0.9 / pk if pk > 0.9 else a
    d = Decoder(samprate=16000)
    for wd, ph in (extra_words or {}).items():
        if d.lookup_word(wd) is None: d.add_word(wd, ph, True)
    words, owner = [], []
    for li, l in enumerate(lines):
        for wd in re.findall(r"[a-z']+", l.lower().replace("’", "'").replace("-", " ")):
            wd = wd.strip("'")
            if not wd: continue
            assert d.lookup_word(wd) is not None, ("sozlukte yok", wd)
            words.append(wd); owner.append(li)
    d.set_align_text(" ".join(words)); d.start_utt(); d.process_raw(data, full_utt=True); d.end_utt()
    segs = [(s.start_frame / 100, s.end_frame / 100) for s in d.seg() if not s.word.startswith(("<", "(", "["))]
    assert len(segs) == len(words), (path, len(segs), len(words))
    out, fade = [], int(0.01 * SR)
    for li in range(len(lines)):
        idx = [i for i, o in enumerate(owner) if o == li]
        t0 = max(0, segs[idx[0]][0] - 0.10) if li == 0 else (segs[idx[0] - 1][1] + segs[idx[0]][0]) / 2
        t1 = len(a) / SR if li == len(lines) - 1 else (segs[idx[-1]][1] + segs[idx[-1] + 1][0]) / 2
        gap_after = None if li == len(lines) - 1 else round(segs[idx[-1] + 1][0] - segs[idx[-1]][1], 2)
        x = a[int(t0 * SR):int(t1 * SR)].copy()
        x[:fade] *= np.linspace(0, 1, fade); x[-fade:] *= np.linspace(1, 0, fade)
        nz = np.where(np.abs(x) > 0.01)[0]
        x = x[: min(len(x), nz[-1] + int(0.15 * SR))]
        out.append(x); print(f"  {os.path.basename(path)} replik {li + 1}: {len(x) / SR:5.2f} sn, sonraki replige bosluk: {gap_after}")
    return out
