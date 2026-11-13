#!/usr/bin/env python3
"""Digital Story 8 - Scene 2 "The Radio Check" montaj scripti.

Calistirma:  python3 build_scene2.py
Cikti:       DigitalStory8_Scene2.mp4  +  CEKIM_LISTESI_Scene2.md
(Ortak yardimci fonksiyonlar build_scene1.py icinden alinir.)
"""
import os, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw
import build_scene1 as b
from build_scene1 import W, H, FPS, SR, GREEN, FF, font, ease, fade_text

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# (agiz acik kare, agiz+goz kapali kare, yuz kutusu)
b.FACES.update({
    "wide2": ("backgrounds/radio_room_wide.png", "backgrounds/radio_room_wide_closed.png", (215, 175, 445, 395)),
    "reyhan2": ("shots/s2_reyhan.png", "shots/s2_reyhan_closed.png", (560, 150, 800, 420)),
    "alex2": ("shots/s2_alex.png", "shots/s2_alex_closed.png", (400, 170, 660, 430)),
    "maya2": ("shots/s2_maya.png", "shots/s2_maya_closed.png", (280, 180, 540, 450)),
    "noah2": ("shots/s2_noah.png", "shots/s2_noah_closed.png", (640, 180, 950, 450), 3),
})

LINES = {
    "1": ("Ms Ellis", "audio/s2_01_ellis.wav", "All right, team. Time for our radio test. Let’s hear how your voices sound through the system."),
    "2": ("Reyhan", "audio/s2_02_reyhan.wav", "Hello from Cumhuriyet Upper Secondary School. We’re ready for our first voyage into space."),
    "3": ("Ms Ellis", "audio/s2_03_ellis.wav", "Loud and clear. Next?"),
    "4": ("Alex", "audio/s2_04_alex.wav", "This is Alex. I’m a volunteer for the tech team. The waveform on the screen looks perfect, and the y-axis is stable too."),
    "5": ("Maya", "audio/s2_05_maya.wav", "Hi, this is Maya. I’m watching the monitor. It feels like the whole world is listening."),
    "6": ("Noah", "audio/s2_06_noah.wav", "Hello, astronaut crew. We’re checking our oxygen levels for this exploration… just kidding!"),
    "7": ("Reyhan", "audio/s2_07_reyhan.wav", "You sound like a real scientist, Noah. I think that’s your X-factor."),
    "8": ("Ms Ellis", "audio/s2_08_ellis.wav", "Nice energy, everyone. Let’s wrap up with one more test, just in case we need plan Z."),
    "9": ("Alex", "audio/s2_09_alex.wav", "Sure. This is Cumhuriyet Upper Secondary School signing off until next year. See you in zero-gravity, maybe."),
}
FULL = (0, 0, 1, 1)
ELLIS = (0.0, 0.04, 0.62, 0.66)      # genel plandan Ms Ellis'e kadraj
KIDS = (0.45, 0.16, 1.0, 0.71)       # genel plandan ogrencilere kadraj
# (gorsel, crop0, crop1, replik, kural, aciklama, animasyon)
PLAN = [
    ("(grafik)", FULL, FULL, None, 2.4, "SAHNE GEÇİŞ KARESİ: “Scene 2 – The Radio Check” (seslendirilmez).", "card"),
    ("backgrounds/radio_room_wide.png", FULL, FULL, None, 1.5, "Okulun radyo/stüdyo alanı: mikrofon, kulaklık, bilgisayar, ses kontrol ünitesi, bağlantı ekranı. Kısa sistem açılış sesi.", "wide2"),
    ("backgrounds/radio_room_wide.png", FULL, ELLIS, "1", "line", "Ms Ellis sistemi hazırlar, öğrencilere işaret verir (ağız hareketi + göz kırpma).", "wide2"),
    ("shots/s2_reyhan.png", FULL, (0.06, 0.0, 0.96, 0.9), "2", ("head", 3.4), "Reyhan mikrofona konuşur (ağız hareketi + göz kırpma).", "reyhan2"),
    ("(grafik: ekran)", FULL, FULL, "2", "tail", "Ekrandaki ses göstergesi Reyhan’ın sesine tepki verir.", "monitor"),
    ("backgrounds/radio_room_wide.png", ELLIS, ELLIS, None, 1.0, "Reyhan’ın sözü tamamen biter; Ms Ellis kulaklıktan dinler, kısa duraklama.", "wide2"),
    ("backgrounds/radio_room_wide.png", ELLIS, (0.02, 0.06, 0.60, 0.64), "3", "line", "Ms Ellis onay verir.", "wide2"),
    ("shots/s2_alex.png", FULL, (0.08, 0.02, 0.98, 0.92), "4", ("head", 3.7), "Alex mikrofona geçer, ekranı işaret eder (ağız hareketi + göz kırpma).", "alex2"),
    ("(grafik: ekran)", FULL, FULL, "4", "tail", "Ekranda sesle hareket eden dalga formu ve y ekseni birlikte görünür.", "monitor"),
    ("shots/s2_maya.png", FULL, (0.0, 0.05, 0.9, 0.95), "5", "line", "Maya monitörü takip ederek konuşur (ağız hareketi + göz kırpma).", "maya2"),
    ("shots/s2_noah.png", (0.05, 0.05, 0.95, 0.95), (0.18, 0.12, 0.98, 0.92), "6", "line", "Noah ciddi anons taklidi yapar, son sözünde şakacı gülümser. Gerçek oksijen ölçümü yok.", "noah2"),
    ("backgrounds/radio_room_wide.png", KIDS, KIDS, None, 1.1, "Arkadaşları hafifçe gülümser.", None),
    ("shots/s2_reyhan.png", (0.06, 0.0, 0.96, 0.9), (0.12, 0.02, 0.92, 0.82), "7", "line", "Reyhan Noah’a dönerek konuşur (ağız hareketi + göz kırpma).", "reyhan2"),
    ("backgrounds/radio_room_wide.png", ELLIS, FULL, "8", "line", "Ms Ellis son bir test ister; kamera odanın geneline açılır.", "wide2"),
    ("shots/s2_alex.png", (0.08, 0.02, 0.98, 0.92), FULL, "9", "line", "Alex yeniden mikrofona yaklaşır; hafif şakacı prova anonsu.", "alex2"),
    ("backgrounds/radio_room_wide.png", KIDS, FULL, None, 1.6, "Odanın genel görünümü; düz kararma (bir yıl geçmiş gibi geçiş yapılmaz).", None),
]
GAP = 0.40


def beep():
    """Kisa, hafif sistem acilis sesi (replikleri ortmez)."""
    t = np.arange(int(0.5 * SR)) / SR
    x = 0.10 * np.sin(2 * np.pi * 880 * t) * np.exp(-9 * t) + 0.08 * np.sin(2 * np.pi * 1320 * np.maximum(t - 0.14, 0)) * np.exp(-9 * np.maximum(t - 0.14, 0)) * (t > 0.14)
    return x.astype(np.float32)


def monitor_frame(mix, t_abs):
    """Baglanti ekrani: sesle hareket eden dalga formu + y ekseni + seviye gostergesi."""
    im = Image.new("RGB", (W, H), (9, 22, 26))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((40, 40, W - 40, H - 40), radius=28, outline=(60, 70, 72), width=10)
    x0, x1, yc, amp = 190, 1040, 370, 215
    for k in range(-2, 3):  # yatay kilavuz cizgileri + y ekseni isaretleri
        y = yc - k * amp / 2
        d.line((x0, y, x1, y), fill=(26, 58, 60), width=2)
        d.line((x0 - 12, y, x0, y), fill=(230, 240, 235), width=3)
        lab = f"{k / 2:+.1f}" if k else "0"
        d.text((x0 - 24 - d.textlength(lab, font=font(24, False)), y - 14), lab, font=font(24, False), fill=(200, 215, 210))
    d.line((x0, yc - amp - 25, x0, yc + amp + 25), fill=(230, 240, 235), width=4)   # y ekseni
    d.line((x0, yc + amp + 25, x1, yc + amp + 25), fill=(230, 240, 235), width=4)   # x ekseni
    d.text((x0 - 12, yc - amp - 70), "y", font=font(36), fill=GREEN)
    d.text((x1 - 10, yc + amp + 34), "x", font=font(36), fill=GREEN)
    # dalga formu: son 1.6 sn
    n = int(1.6 * SR); i1 = int(t_abs * SR); seg = mix[max(0, i1 - n):i1]
    if len(seg) < n: seg = np.concatenate([np.zeros(n - len(seg), np.float32), seg])
    cols = 425; seg = seg[: n // cols * cols].reshape(cols, -1)
    hi = np.clip(seg.max(1) * 1.6, -1, 1); lo = np.clip(seg.min(1) * 1.6, -1, 1)
    pts = []
    for c in range(cols):
        x = x0 + 4 + c * (x1 - x0 - 8) / cols
        d.line((x, yc - hi[c] * amp, x, yc - lo[c] * amp), fill=GREEN, width=2)
        pts.append((x, yc - (hi[c] + lo[c]) / 2 * amp))
    d.line(pts, fill=(170, 255, 200), width=2)
    # seviye gostergesi
    rms = float(np.sqrt(np.mean(mix[max(0, i1 - SR // 12):i1 + 1] ** 2))) if i1 > 0 else 0
    lvl = min(1, rms * 5.5); bx0, bx1, by0, by1 = 1110, 1170, 150, 610
    d.rounded_rectangle((bx0 - 6, by0 - 6, bx1 + 6, by1 + 6), radius=10, outline=(200, 215, 210), width=3)
    segs = 16
    for k in range(segs):
        on = k / segs < lvl
        col = (GREEN if k < 11 else (255, 200, 60) if k < 14 else (235, 80, 70)) if on else (24, 48, 50)
        yb = by1 - k * (by1 - by0) / segs
        d.rectangle((bx0, yb - (by1 - by0) / segs + 5, bx1, yb), fill=col)
    d.text((190, 78), "SOUND CHECK", font=font(34), fill=(230, 240, 235))
    blink = (t_abs * 2) % 1 < 0.6
    d.ellipse((1010, 86, 1034, 110), fill=(235, 80, 70) if blink else (70, 30, 30))
    d.text((1046, 80), "MIC", font=font(30), fill=(230, 240, 235))
    return im


def build():
    audio = {k: b.load_wav(v[1]) for k, v in LINES.items()}
    segs, t, placed = [], 0.0, {}
    for img, c0, c1, lid, rule, note, anim in PLAN:
        if lid is None:
            dur = rule
        else:
            L = len(audio[lid]) / SR
            if rule == "line":
                placed[lid] = t; dur = L + GAP
            elif rule == "tail":
                dur = placed[lid] + L + GAP - t
            else:
                placed[lid] = t; dur = rule[1]
        segs.append(dict(img=img, c0=c0, c1=c1, lid=lid, start=t, dur=dur, note=note, anim=anim, idx=len(segs)))
        t += dur
    total = t

    mix = np.zeros(int((total + 0.5) * SR), dtype=np.float32)
    voice = np.zeros_like(mix)
    for lid, st in placed.items():
        a = audio[lid]; i = int(st * SR); voice[i:i + len(a)] += a
    bp = beep(); i = int(2.7 * SR); mix[i:i + len(bp)] += bp
    mix += voice; mix = np.clip(mix, -0.98, 0.98)
    with wave.open("/tmp/scene2_mix.wav", "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())

    tracks = {k: b.mouth_track(a) for k, a in audio.items()}
    cache = {}

    def source(seg, tt):
        an = seg["anim"]
        if an == "card": return b.card_frame(tt, seg["dur"], 2, "The Radio Check")
        if an == "monitor": return monitor_frame(voice, seg["start"] + tt)
        if an in b.FACES:
            fs = b.face_set(an); closed = 1
            if seg["lid"]:
                i = int(round((seg["start"] + tt - placed[seg["lid"]]) * FPS)); tr = tracks[seg["lid"]]
                if 0 <= i < len(tr) and tr[i]: closed = 0
            return fs[(closed, 1 if b.blink_at(seg["start"] + tt, seg["idx"]) else 0)]
        if seg["img"] not in cache:
            cache[seg["img"]] = Image.open(seg["img"]).convert("RGB")
        return cache[seg["img"]]

    def frame_of(seg, tt):
        tt = min(tt, seg["dur"] - 1e-3)
        im = source(seg, tt)
        p = ease(tt / seg["dur"])
        c = [a + (bb - a) * p for a, bb in zip(seg["c0"], seg["c1"])]
        if c == [0, 0, 1, 1] and im.size == (W, H): return im
        return im.resize((W, H), Image.BICUBIC, box=(c[0] * im.width, c[1] * im.height, c[2] * im.width, c[3] * im.height))

    out = "DigitalStory8_Scene2.mp4"
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-i", "/tmp/scene2_mix.wav", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                         stdin=subprocess.PIPE)
    XF = 0.3; black = Image.new("RGB", (W, H))
    for i in range(int(total * FPS)):
        tt = i / FPS
        k = max(j for j, s in enumerate(segs) if s["start"] <= tt + 1e-9)
        s = segs[k]; local = tt - s["start"]
        fr = frame_of(s, local)
        if k > 0 and local < XF and segs[k - 1]["img"] != s["img"]:
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
        if s["lid"]:
            who = LINES[s["lid"]][0]; snd = os.path.basename(LINES[s["lid"]][1])
            line = "(önceki replik sürer)" if s["lid"] in said else LINES[s["lid"]][2]; said.add(s["lid"])
        if i == 2: snd = "sistem açılış sesi"
        rows.append(f"| {i} | {tc(s['start'])}–{tc(s['start'] + s['dur'])} | `{s['img']}` | {s['note']} | {who} | {line} | {snd} |")
    md = f"""# Digital Story 8 – Scene 2 "The Radio Check" – Çekim Listesi

**Toplam süre:** {tc(total)} (sahne geçiş karesi dâhil)
**Format:** 1280×720, 25 fps, MP4 (H.264 + AAC)

Karakterler Scene 1 ile aynıdır (`characters/character_sheet.png`). Kıyafetlerin görünümü korunur.

## Sabit mekân

Referans görsel: `backgrounds/radio_room_wide.png` — okulun radyo/stüdyo alanı: masa mikrofonu, kulaklık, dizüstü bilgisayar, ses kontrol ünitesi, duvarda bağlantı ekranı, akustik paneller.

## Çekim listesi

{chr(10).join(rows)}

## Talimat belgesindeki kurallara uyum

- "Scene 2 – The Radio Check" ayrı bir geçiş karesinde gösterilir, seslendirilmez.
- Bu sahne yerel ses provasıdır; astronotla bağlantı kurulmaz, astronot görünmez.
- Reyhan konuşurken ekrandaki ses göstergesi tepki verir.
- Reyhan’ın sözü tamamen bittikten sonra Ms Ellis kısa bir duraklamayla onay verir.
- Alex’in repliğinde dalga formu ve y ekseni ekranda birlikte görünür.
- Gerçek bir oksijen ölçümü gösterilmez.
- Sahne sonunda düz kararma kullanılır; bir yıl geçmiş gibi geçiş yapılmaz.
- Teknik ses yalnızca sahne başında, replik yokken duyulur.
"""
    open("CEKIM_LISTESI_Scene2.md", "w", encoding="utf-8").write(md)
    print("OK", out, "sure:", tc(total))
    for s in segs: print(f"{s['start']:6.2f} {s['dur']:5.2f} {s['img']} {s['lid']}")


if __name__ == "__main__":
    build()
