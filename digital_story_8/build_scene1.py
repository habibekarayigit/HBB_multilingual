#!/usr/bin/env python3
"""Digital Story 8 - Scene 1 "The Announcement" montaj scripti.

Calistirma:  python3 build_scene1.py
Gerekenler:  pip install numpy pillow imageio-ffmpeg matplotlib
Cikti:       DigitalStory8_Scene1.mp4  +  CEKIM_LISTESI_Scene1.md
"""
import os, subprocess, wave, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio_ffmpeg, matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
FF = imageio_ffmpeg.get_ffmpeg_exe()
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts/ttf")
W, H, FPS, SR = 1280, 720, 25, 44100
GREEN = (46, 204, 113)
os.makedirs("cards", exist_ok=True)


def font(size, bold=True):
    return ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def ctext(d, cx, y, text, f, fill=(255, 255, 255), shadow=True):
    w = d.textlength(text, font=f)
    if shadow:
        d.text((cx - w / 2 + 3, y + 3), text, font=f, fill=(0, 0, 0))
    d.text((cx - w / 2, y), text, font=f, fill=fill)


# ---------------------------------------------------------------- logo / kartlar
def make_logo(size=260):
    s = size * 4
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((s * .12, s * .12, s * .88, s * .88), fill=(20, 40, 90, 255), outline=(255, 255, 255, 255), width=s // 40)
    # yorunge halkasi
    ring = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((s * .02, s * .36, s * .98, s * .64), outline=GREEN + (255,), width=s // 45)
    ring = ring.rotate(-22, resample=Image.BICUBIC)
    im.alpha_composite(ring)
    # play ucgeni
    d = ImageDraw.Draw(im)
    d.polygon([(s * .42, s * .33), (s * .42, s * .67), (s * .70, s * .50)], fill=(255, 255, 255, 255))
    # kucuk uydu noktasi
    d.ellipse((s * .86, s * .22, s * .95, s * .31), fill=(255, 200, 60, 255))
    im = im.resize((size, size), Image.LANCZOS)
    im.save("cards/logo_digital_story.png")
    return im


def make_cover():
    bg = Image.open("shots/cover_bg.png").convert("RGB").resize((W, H), Image.LANCZOS)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rounded_rectangle((170, 150, W - 170, 590), radius=36, fill=(8, 16, 44, 205), outline=GREEN + (255,), width=4)
    bg = Image.alpha_composite(bg.convert("RGBA"), ov)
    logo = make_logo(150)
    bg.alpha_composite(logo, (W // 2 - 75, 168))
    d = ImageDraw.Draw(bg)
    ctext(d, W // 2, 332, "THEME 8", font(46), GREEN)
    ctext(d, W // 2, 396, "LIFE IN THE UNIVERSE & THE FUTURE", font(38))
    d.line((W // 2 - 160, 468, W // 2 + 160, 468), fill=GREEN, width=4)
    ctext(d, W // 2, 490, "DIGITAL STORY 8", font(58), GREEN)
    bg.convert("RGB").save("cards/cover.png")


def make_scene_card(num, title):
    bg = Image.open("shots/cover_bg.png").convert("RGB").resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(8))
    bg = Image.blend(bg, Image.new("RGB", (W, H), (6, 12, 34)), 0.55)
    d = ImageDraw.Draw(bg)
    ctext(d, W // 2, 250, f"Scene {num}", font(64), GREEN)
    d.line((W // 2 - 140, 345, W // 2 + 140, 345), fill=(255, 255, 255), width=4)
    ctext(d, W // 2, 375, title, font(76))
    bg.save(f"cards/scene{num}_card.png")


def make_diagram():
    im = Image.open("shots/diagram_base.png").convert("RGB")
    d = ImageDraw.Draw(im)
    f = font(40)
    ctext(d, 243, 590, "School", f)
    ctext(d, 698, 600, "ARISS ground team", f)
    ctext(d, 1133, 340, "ISS", f)
    ctext(d, 470, 525, "radio link", font(30, False), GREEN, shadow=False)
    im.save("shots/diagram.png")



# ---------------------------------------------------------------- animasyon yardimcilari
from scipy import ndimage

# (acik agizli kare, kapali agiz+kapali goz karesi, yuz kutusu)
FACES = {
    "ellis": ("shots/ellis_medium.png", "shots/ellis_medium_closed.png", (600, 180, 810, 400)),
    "reyhan": ("shots/reyhan.png", "shots/reyhan_closed.png", (660, 180, 920, 500)),
    "alex": ("shots/alex.png", "shots/alex_closed.png", (620, 150, 860, 420)),
    "maya": ("shots/maya.png", "shots/maya_closed.png", (560, 190, 820, 450)),
    "noah": ("shots/noah.png", "shots/noah_closed.png", (600, 200, 880, 470)),
}
_face_cache = {}


def face_set(key):
    """4 kare dondurur: [agiz acik/kapali][goz acik/kapali]. Yalnizca agiz ve goz bolgesi degisir."""
    if key in _face_cache:
        return _face_cache[key]
    a, b, box = FACES[key]
    A = Image.open(a).convert("RGB"); B = Image.open(b).convert("RGB").resize(A.size, Image.LANCZOS)
    d = np.abs(np.asarray(A, float) - np.asarray(B, float)).mean(2)
    m = np.zeros_like(d, bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = d[y0:y1, x0:x1] > 30
    m = ndimage.binary_dilation(m, iterations=6)
    lab, n = ndimage.label(m)
    comps = []
    for i in range(1, n + 1):
        area = (lab == i).sum()
        if area > 500:
            comps.append((ndimage.center_of_mass(lab == i)[0], i))
    comps.sort()
    cys = [c[0] for c in comps]
    k = int(np.argmax(np.diff(cys))) + 1  # en buyuk dikey bosluk: gozler | agiz
    eye = np.isin(lab, [c[1] for c in comps[:k]]); mouth = np.isin(lab, [c[1] for c in comps[k:]])
    def soft(mask):
        mask = ndimage.binary_dilation(mask, iterations=5)
        return Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
    em, mm = soft(eye), soft(mouth)
    out = {}
    for mc in (0, 1):
        for ec in (0, 1):
            im = A.copy()
            if mc: im.paste(B, (0, 0), mm)
            if ec: im.paste(B, (0, 0), em)
            out[(mc, ec)] = im
    _face_cache[key] = out
    return out


def wide_set():
    """Genel plan: ogretmen konusurken agzi acik kare (yalnizca bas bolgesi degisir)."""
    if "wide" in _face_cache:
        return _face_cache["wide"]
    A = Image.open("backgrounds/classroom_wide.png").convert("RGB")
    B = Image.open("backgrounds/classroom_wide_talk.png").convert("RGB").resize(A.size, Image.LANCZOS)
    mk = Image.new("L", A.size, 0)
    ImageDraw.Draw(mk).ellipse((138, 262, 222, 348), fill=255)
    mk = mk.filter(ImageFilter.GaussianBlur(4))
    T = A.copy(); T.paste(B, (0, 0), mk)
    _face_cache["wide"] = {(1, 0): A, (0, 0): T, (1, 1): A, (0, 1): T}
    return _face_cache["wide"]


def mouth_track(a):
    """Ses zarfindan kare kare agiz durumu (True = acik)."""
    hop = SR // FPS
    n = len(a) // hop + 1
    env = np.array([np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) if len(a[i * hop:(i + 1) * hop]) else 0 for i in range(n)])
    thr = 0.30 * np.percentile(env, 90)
    op = env > thr
    run = 0
    for i in range(n):  # uzun acik kalmasin: hece hissi icin ara ara kapat
        run = run + 1 if op[i] else 0
        if run > 5:
            op[i] = False; run = 0
    return op


def blink_at(t, seed):
    """~3.4 sn'de bir, 3 karelik goz kirpma."""
    period = 3.4 + (seed % 5) * 0.3
    ph = (t + seed * 0.77) % period
    return ph < 0.12


def ease(x):
    x = min(1, max(0, x)); return x * x * (3 - 2 * x)


def fade_text(base, cx, y, text, f, fill, alpha, dy=0):
    if alpha <= 0: return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = d.textlength(text, font=f)
    d.text((cx - w / 2 + 3, y + dy + 3), text, font=f, fill=(0, 0, 0, int(200 * alpha)))
    d.text((cx - w / 2, y + dy), text, font=f, fill=fill + (int(255 * alpha),))
    base.alpha_composite(layer)


_cov = {}
def cover_frame(t, dur):
    if not _cov:
        _cov["bg"] = Image.open("shots/cover_bg.png").convert("RGB")
        _cov["logo"] = make_logo(150)
    bg = _cov["bg"]; z = 0.06 * ease(t / dur)
    im = bg.resize((W, H), Image.BICUBIC, box=(bg.width * z / 2, bg.height * z / 2, bg.width * (1 - z / 2), bg.height * (1 - z / 2))).convert("RGBA")
    a = ease((t - 0.2) / 0.5)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle((170, 150, W - 170, 590), radius=36, fill=(8, 16, 44, int(205 * a)), outline=GREEN + (int(255 * a),), width=4)
    im.alpha_composite(ov)
    s = ease((t - 0.5) / 0.45)
    if s > 0:  # logo buyuyerek ve donerek gelir
        size = max(2, int(150 * (0.3 + 0.7 * s) * (1 + 0.12 * np.sin(np.pi * s))))
        lg = _cov["logo"].rotate((1 - s) * 180, resample=Image.BICUBIC).resize((size, size), Image.LANCZOS)
        im.alpha_composite(lg, (W // 2 - size // 2, 243 - size // 2))
    for txt, y, f, col, t0 in (("THEME 8", 332, font(46), GREEN, 1.0), ("LIFE IN THE UNIVERSE & THE FUTURE", 396, font(38), (255, 255, 255), 1.35), ("DIGITAL STORY 8", 490, font(58), GREEN, 1.9)):
        p = ease((t - t0) / 0.45); fade_text(im, W // 2, y, txt, f, col, p, dy=int(18 * (1 - p)))
    p = ease((t - 1.7) / 0.4)
    if p > 0: ImageDraw.Draw(im).line((W // 2 - 160 * p, 468, W // 2 + 160 * p, 468), fill=GREEN, width=4)
    return im.convert("RGB")


def card_frame(t, dur, num=1, title="The Announcement"):
    if "card" not in _cov:
        bg = Image.open("shots/cover_bg.png").convert("RGB").resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(8))
        _cov["card"] = Image.blend(bg, Image.new("RGB", (W, H), (6, 12, 34)), 0.55).convert("RGBA")
    im = _cov["card"].copy()
    p = ease((t - 0.15) / 0.4); fade_text(im, W // 2, 250, f"Scene {num}", font(64), GREEN, p, dy=int(-20 * (1 - p)))
    p = ease((t - 0.4) / 0.4)
    if p > 0: ImageDraw.Draw(im).line((W // 2 - 140 * p, 345, W // 2 + 140 * p, 345), fill=(255, 255, 255), width=4)
    p = ease((t - 0.6) / 0.45); fade_text(im, W // 2, 375, title, font(76), (255, 255, 255), p, dy=int(20 * (1 - p)))
    return im.convert("RGB")


def diagram_frame(t, dur):
    """Etiketler sirayla belirir; yesil sinyal noktalari okul -> yer ekibi -> ISS yolunda ileri geri akar."""
    if "dia" not in _cov:
        _cov["dia"] = Image.open("shots/diagram_base.png").convert("RGBA")
    im = _cov["dia"].copy()
    fade_text(im, 243, 590, "School", font(40), (255, 255, 255), ease((t - 0.4) / 0.5))
    fade_text(im, 1133, 340, "ISS", font(40), (255, 255, 255), ease((t - 3.2) / 0.5))
    fade_text(im, 470, 525, "radio link", font(30, False), GREEN, ease((t - 6.0) / 0.5))
    fade_text(im, 698, 600, "ARISS ground team", font(40), (255, 255, 255), ease((t - 9.6) / 0.5))
    P = [np.array(p, float) for p in ((330, 450), (500, 330), (640, 400))], [np.array(p, float) for p in ((760, 380), (900, 180), (1040, 215))]
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    for seg_i, (p0, p1, p2) in enumerate(P):
        a = ease((t - (5.5 if seg_i == 0 else 6.5)) / 0.8)
        if a <= 0: continue
        for k in range(4):
            u = ((t * 0.45 + k / 4) % 1.0)
            for uu, col in ((u, GREEN), (1 - u, (255, 200, 60))):  # gidis (yesil) ve donus (sari)
                q = (1 - uu) ** 2 * p0 + 2 * (1 - uu) * uu * p1 + uu ** 2 * p2
                al = int(255 * a * np.sin(np.pi * uu))
                d.ellipse((q[0] - 14, q[1] - 14, q[0] + 14, q[1] + 14), fill=col + (al // 4,))
                d.ellipse((q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7), fill=col + (al,))
    im.alpha_composite(layer)
    return im.convert("RGB")

# ---------------------------------------------------------------- ses
def load_wav(path):
    tmp = "/tmp/_conv.wav"
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", path, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", tmp], check=True)
    with wave.open(tmp) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    # bas/son sessizligi kirp
    idx = np.where(np.abs(a) > 0.01)[0]
    if len(idx):
        a = a[max(0, idx[0] - int(.05 * SR)): min(len(a), idx[-1] + int(.15 * SR))]
    # ses seviyesi esitle (RMS -20 dBFS, tepe 0.9)
    rms = np.sqrt(np.mean(a[np.abs(a) > 0.01] ** 2))
    a = a * (0.1 / rms)
    pk = np.abs(a).max()
    if pk > 0.9:
        a = a * 0.9 / pk
    return a


def make_jingle(dur=3.8):
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)

    def bell(freq, start, amp=0.5, decay=2.2):
        tt = t - start
        m = tt >= 0
        env = np.zeros_like(t)
        env[m] = np.exp(-decay * tt[m]) * np.minimum(1, tt[m] / 0.005)
        return amp * env * (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2 * t) + 0.12 * np.sin(2 * np.pi * freq * 3.01 * t))

    notes = [(523.25, 0.0), (659.25, 0.22), (783.99, 0.44), (1046.5, 0.66), (987.77, 1.0), (1174.66, 1.22), (1567.98, 1.5)]
    for f, s in notes:
        out += bell(f, s, 0.38)
    for f in (261.63, 392.0, 523.25, 659.25):  # son akor (pad)
        tt = t - 1.5
        env = np.where(tt >= 0, np.minimum(1, tt / 0.3) * np.exp(-1.3 * np.maximum(tt, 0)), 0)
        out += 0.12 * env * np.sin(2 * np.pi * f * t)
    fade = int(0.5 * SR)
    out[-fade:] *= np.linspace(1, 0, fade)
    out *= 0.7 / np.abs(out).max()
    with wave.open("audio/jingle.wav", "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((out * 32767).astype(np.int16).tobytes())
    return out


# ---------------------------------------------------------------- zaman cizelgesi
# crop = (x0, y0, x1, y1) 0-1 oraninda; baslangic -> bitis (yavas kamera hareketi)
FULL = (0, 0, 1, 1)
LINES = {
    "1a": ("Ms Ellis", "audio/s1_01a_ellis.wav", "Morning, everyone! I have some news that might surprise you."),
    "1b": ("Ms Ellis", "audio/s1_01b_ellis.wav", "ARISS has chosen our school for live contact with the International Space Station (ISS)."),
    "2": ("Reyhan", "audio/s1_02_reyhan.wav", "ARISS? What’s that?"),
    "3": ("Ms Ellis", "audio/s1_03_ellis.wav", "It is a project where students can talk to astronauts on the ISS through a radio link supported by the Amateur Radio on the International Space Station (ARISS) ground team."),
    "4": ("Alex", "audio/s1_04_alex.wav", "You mean we’ll really speak to someone in space?"),
    "5": ("Ms Ellis", "audio/s1_05_ellis.wav", "Exactly. We will prepare our questions and practise before the big day."),
    "6": ("Maya", "audio/s1_06_maya.wav", "Wow, it sounds like science fiction, but maybe not!"),
    "7": ("Noah", "audio/s1_07_noah.wav", "So our voices will actually be transmitted into space. That’s amazing."),
    "8": ("Ms Ellis", "audio/s1_08_ellis.wav", "Yes, and the ARISS team will help us set everything up. Let’s get started."),
}
ELLIS = (0.06, 0.13, 0.93, 1.0)  # ust kenardaki kesik cerceveyi disarida birakir
# (gorsel, crop0, crop1, replik id | None, sure/kural, aciklama)
PLAN = [
    ("cards/cover.png", FULL, FULL, None, 4.6, "KAPAK: Theme 8 + Digital Story 8. Logo dönerek gelir, yazılar sırayla belirir. Jingle çalar ve replikten önce biter.", "cover"),
    ("cards/scene1_card.png", FULL, FULL, None, 2.4, "SAHNE GEÇİŞ KARESİ: “Scene 1 – The Announcement” yazısı kayarak belirir (seslendirilmez).", "card"),
    ("backgrounds/classroom_wide.png", FULL, FULL, None, 1.6, "Sınıfın genel görünümü; şeref köşesi, pano ve uzay posteri görünür.", "wide"),
    ("backgrounds/classroom_wide.png", FULL, (0.0, 0.12, 0.62, 0.74), "1a", "line", "Genel plandan Ms Ellis’e yaklaşılır; Ms Ellis konuşur (ağız hareketi), öğrenciler dinler.", "wide"),
    ("shots/ellis_medium.png", ELLIS, (0.12, 0.17, 0.90, 0.95), "1b", ("head", 3.9), "Ms Ellis orta plan, duyuruyu yapar (ağız hareketi + göz kırpma).", "ellis"),
    ("shots/iss.png", (0.0, 0.0, 0.88, 0.88), (0.12, 0.12, 1.0, 1.0), "1b", "tail", "ISS’den söz edilince ISS görseli gösterilir; kamera yörünge boyunca kayar.", None),
    ("shots/reyhan.png", FULL, (0.05, 0.03, 0.95, 0.93), "2", "line", "Reyhan yakın plan, meraklı/şaşkın soru sorar (ağız hareketi + göz kırpma).", "reyhan"),
    ("shots/diagram.png", FULL, (0.03, 0.03, 0.97, 0.97), "3", "line", "Okul – ARISS yer ekibi – ISS radyo bağlantısı şeması: etiketler sırayla belirir, sinyal noktaları iki yönde akar.", "diagram"),
    ("shots/alex.png", FULL, (0.0, 0.0, 0.9, 0.9), "4", "line", "Alex öğretmene bakarak heyecanla sorar (ağız hareketi + göz kırpma).", "alex"),
    ("shots/ellis_medium.png", (0.12, 0.17, 0.90, 0.95), ELLIS, "5", "line", "Ms Ellis orta plan, onaylar ve planı anlatır (ağız hareketi + göz kırpma).", "ellis"),
    ("shots/maya.png", FULL, (0.08, 0.02, 0.98, 0.92), "6", "line", "Maya yakın plan, hayranlıkla konuşur (ağız hareketi + göz kırpma).", "maya"),
    ("shots/noah.png", FULL, (0.02, 0.05, 0.92, 0.95), "7", "line", "Noah yakın plan, gökyüzünü işaret eder (ağız hareketi + göz kırpma).", "noah"),
    ("backgrounds/classroom_wide.png", (0.0, 0.12, 0.62, 0.74), FULL, "8", "line", "Ms Ellis konuşur (ağız hareketi); kamera sınıfın geneline açılır.", "wide"),
    ("shots/notebooks_fixed.png", FULL, (0.30, 0.30, 1.0, 1.0), None, 3.2, "Öğrenciler defterlerini açar; kamera öğrencilere yaklaşır, radyo testi sahnesine geçiş (kararma).", None),
]
GAP = 0.40  # replikler arasi nefes payi


def build():
    make_cover(); make_scene_card(1, "The Announcement"); make_diagram()  # sabit onizleme kareleri
    jingle = make_jingle()
    audio = {k: load_wav(v[1]) for k, v in LINES.items()}

    segs, t, placed = [], 0.0, {}
    for img, c0, c1, lid, rule, note, anim in PLAN:
        if lid is None:
            dur = rule
        else:
            L = len(audio[lid]) / SR
            if rule == "line":
                placed[lid] = t; dur = L + GAP
            elif rule[0] == "head":
                placed[lid] = t; dur = rule[1]
            if rule == "tail":
                dur = placed[lid] + L + GAP - t
        segs.append(dict(img=img, c0=c0, c1=c1, lid=lid, start=t, dur=dur, note=note, anim=anim, idx=len(segs)))
        t += dur
    total = t

    # --- ses miksi
    mix = np.zeros(int((total + 0.5) * SR), dtype=np.float32)
    mix[int(0.3 * SR): int(0.3 * SR) + len(jingle)] += jingle
    for lid, st in placed.items():
        a = audio[lid]; i = int(st * SR)
        mix[i:i + len(a)] += a
    mix = np.clip(mix, -0.98, 0.98)
    with wave.open("/tmp/scene1_mix.wav", "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())

    # --- goruntu
    cache = {}
    tracks = {k: mouth_track(a) for k, a in audio.items()}

    def source(seg, tt):
        an = seg["anim"]
        if an == "cover": return cover_frame(tt, seg["dur"])
        if an == "card": return card_frame(tt, seg["dur"])
        if an == "diagram": return diagram_frame(tt, seg["dur"])
        if an in FACES or an == "wide":
            fs = face_set(an) if an in FACES else wide_set()
            closed = 1
            if seg["lid"]:
                i = int(round((seg["start"] + tt - placed[seg["lid"]]) * FPS)); tr = tracks[seg["lid"]]
                if 0 <= i < len(tr) and tr[i]: closed = 0
            return fs[(closed, 1 if blink_at(seg["start"] + tt, seg["idx"]) else 0)]
        if seg["img"] not in cache:
            cache[seg["img"]] = Image.open(seg["img"]).convert("RGB")
        return cache[seg["img"]]

    def frame_of(seg, tt):
        tt = min(tt, seg["dur"] - 1e-3)
        im = source(seg, tt)
        p = ease(tt / seg["dur"])
        c = [a + (b - a) * p for a, b in zip(seg["c0"], seg["c1"])]
        if c == [0, 0, 1, 1] and im.size == (W, H): return im
        box = (c[0] * im.width, c[1] * im.height, c[2] * im.width, c[3] * im.height)
        return im.resize((W, H), Image.BICUBIC, box=box)

    out = "DigitalStory8_Scene1.mp4"
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-i", "/tmp/scene1_mix.wav", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                         stdin=subprocess.PIPE)
    XF = 0.3
    black = Image.new("RGB", (W, H))
    n = int(total * FPS)
    for i in range(n):
        tt = i / FPS
        k = max(j for j, s in enumerate(segs) if s["start"] <= tt + 1e-9)
        s = segs[k]; local = tt - s["start"]
        fr = frame_of(s, local)
        if k > 0 and local < XF:  # yumusak gecis
            prev = segs[k - 1]
            fr = Image.blend(frame_of(prev, prev["dur"]), fr, local / XF)
        if tt < 0.4: fr = Image.blend(black, fr, tt / 0.4)
        if total - tt < 0.8: fr = Image.blend(black, fr, max(0, (total - tt) / 0.8))
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()

    # --- cekim listesi
    def tc(x): return f"{int(x // 60):02d}:{x % 60:04.1f}"
    rows = ["| # | Zaman | Görsel dosyası | Ekranda ne var / kamera | Konuşan | Replik (aynen okunur) | Ses |", "|---|---|---|---|---|---|---|"]
    said = set()
    for i, s in enumerate(segs, 1):
        who = line = ""; snd = "—"
        if s["lid"]:
            who = LINES[s["lid"]][0]
            if s["lid"] in said:
                line = "(önceki replik sürer)"
            else:
                line = LINES[s["lid"]][2]; said.add(s["lid"])
            snd = os.path.basename(LINES[s["lid"]][1])
        if i == 1: snd = "jingle.wav"
        rows.append(f"| {i} | {tc(s['start'])}–{tc(s['start'] + s['dur'])} | `{s['img']}` | {s['note']} | {who} | {line} | {snd} |")
    md = open("CEKIM_LISTESI_Scene1.template.md", encoding="utf-8").read().replace("{{TABLO}}", "\n".join(rows)).replace("{{SURE}}", tc(total))
    open("CEKIM_LISTESI_Scene1.md", "w", encoding="utf-8").write(md)
    print("OK", out, "sure:", tc(total))
    for s in segs: print(f"{s['start']:6.2f} {s['dur']:5.2f} {s['img']} {s['lid']}")


if __name__ == "__main__":
    build()
