#!/usr/bin/env python3
"""Digital Story 8 - Scene 4 "The Big Connection".  Calistirma: python3 build_scene4.py"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import signal
import build_scene1 as b
from build_scene1 import W, H, GREEN, SR, ease
from story_engine import render, inset, align_split, auto_box, FULL

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# sirayla replikler: (kimlik, konusan, metin)
S = [
    ("1", "Station", "Cumhuriyet Upper Secondary School, the International Space Station is now in range. Stand by for contact."),
    ("2", "Ms Ellis", "Cumhuriyet Upper Secondary School standing by."),
    ("3", "Hayes", "Cumhuriyet Upper Secondary School, this is Commander Hayes. We hear you loud and clear. You are cleared for contact."),
    ("4", "Ms Ellis", "Thank you, Commander. We’re ready."),
    ("5", "Reyhan", "Hello from the world! We’ve been waiting all week for this call. We’ve prepared some questions about life in space."),
    ("6", "Hayes", "Wonderful! Go ahead with your first question."),
    ("7", "Reyhan", "First, how do you manage oxygen levels and protect yourselves from radiation up there?"),
    ("8", "Hayes", "We monitor the air all the time and use systems to keep oxygen at safe levels. The station also has shielding, and we follow safety rules to reduce radiation risks."),
    ("9", "Reyhan", "If you spill water up there, does it just float around?"),
    ("10", "Hayes", "Yes, it does. Everything moves slowly here. If you heat soup, it doesn’t boil like on Earth, but it makes little floating bubbles."),
    ("11", "Alex", "That sounds fun! What happens if you forget to close something, like a food bag?"),
    ("12", "Hayes", "If that happens, then everything floats away: cookies, crumbs, even pencils… We spend much of our time catching things. Just kidding!"),
    ("13", "Reyhan", "I’d be terrible at that. If I were there, I would lose everything in five minutes."),
    ("14", "Maya", "I read about a future Mars colony last week. If we had a colony on Mars, would kids study online or in real schools?"),
    ("15", "Hayes", "Probably online, but if I had a choice, I wouldn’t trade a real classroom for a screen. You know, you cannot high-five a classmate through space Wi-Fi."),
    ("16", "Noah", "True! So, if we find another planet like Earth, will people move there someday?"),
    ("17", "Hayes", "Maybe one day. Scientists are searching for habitable planets all the time. If we don’t cooperate, we won’t survive long in space."),
    ("18", "Noah", "I wish people didn’t waste so much energy and resources on Earth. Sometimes it feels like we forget how lucky we are."),
    ("19", "Alex", "That’s so true. If everyone helped more, for example, by recycling, saving water, and using less energy, the world would stay cleaner. Maybe we wouldn’t even look for another planet. I wish people would take the future more seriously before problems grow bigger."),
    ("20", "Reyhan", "And I wish I could float like you for a few minutes, just to see the Earth from there."),
    ("21", "Hayes", "It’s beautiful, yes. If people didn’t take care of our planet, we would have to look for another one to live on."),
    ("22", "Reyhan", "Commander, do you ever miss home?"),
    ("23", "Hayes", "Every day. Sometimes I wish I had spent more time with my family before this mission began."),
    ("24", "Ms Ellis", "Thank you, Commander. You have inspired our whole class today."),
    ("25", "Station", "Contact ending in two minutes. Congratulations, students!"),
    ("26", "Alex", "Let’s mark this day down. If we ever fail to believe in our dreams, we’ll remember this day."),
]
TXT = {k: t for k, _, t in S}
# ses kayitlari: dosya -> icindeki replikler (sirayla)
CLIPS = {
    "s4_station.wav": ["1", "25"], "s4_ellis.wav": ["2", "4", "24"],
    "s4_hayes_A.wav": ["3", "6", "8"], "s4_hayes_B.wav": ["10", "12", "15"], "s4_hayes_C.wav": ["17", "21", "23"],
    "s4_reyhan_A.wav": ["5", "7", "9"], "s4_reyhan_B.wav": ["13", "20", "22"],
    "s4_alex.wav": ["11", "19", "26"], "s4_noah.wav": ["16", "18"], "s4_maya.wav": ["14"],
}
EXTRA = {"cumhuriyet": "JH UH M HH UH R IY Y EH T", "wi": "W AY", "fi": "F AY"}


def radio(x):
    """Hafif telsiz rengi (Station ve Hayes radyo uzerinden duyulur); anlasilirlik icin kuru sesle karistirilir."""
    sos = signal.butter(3, [300, 3800], btype="band", fs=SR, output="sos")
    y = 0.75 * signal.sosfilt(sos, x) + 0.25 * x
    return (y * (np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(y ** 2)))).astype(np.float32)


def link_screen(tt, seg, ctx):
    """Baglanti gostergesi: ISS gorseli + yukselen sinyal cubuklari (yazi yok)."""
    if "iss" not in _c:
        _c["iss"] = Image.open("shots/iss.png").convert("RGB").resize((720, 405), Image.LANCZOS)
    im = Image.new("RGB", (W, H), (9, 16, 40)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((40, 40, W - 40, H - 40), radius=28, outline=(60, 72, 110), width=8)
    mk = Image.new("L", (720, 405), 0); ImageDraw.Draw(mk).rounded_rectangle((0, 0, 719, 404), radius=24, fill=255)
    im.paste(_c["iss"], (100, 158), mk)
    t = seg["start"] + tt
    n = 5 if seg.get("steady") else min(5, int(1 + tt / 0.45))
    for k in range(5):
        x = 900 + k * 58; h = 60 + k * 55
        d.rounded_rectangle((x, 520 - h, x + 40, 520), radius=8, fill=GREEN if k < n else (34, 48, 80))
    r = 16 + 8 * (0.5 + 0.5 * np.sin(t * 6))
    d.ellipse((1040 - r, 190 - r, 1040 + r, 190 + r), outline=GREEN, width=4)
    d.ellipse((1030, 180, 1050, 200), fill=GREEN)
    return im


_c = {}


def blueprint(src, dst):
    """Mars kolonisini 'gelecek taslagi' gibi gosterir (mevcut bir yerlesim gibi degil)."""
    g = ImageOps.autocontrast(Image.open(src).convert("L"))
    im = ImageOps.colorize(g, (16, 44, 96), (214, 236, 255)).convert("RGB"); d = ImageDraw.Draw(im)
    for x in range(0, im.width, 86): d.line((x, 0, x, im.height), fill=(120, 170, 230), width=1)
    for y in range(0, im.height, 86): d.line((0, y, im.width, y), fill=(120, 170, 230), width=1)
    for x in range(20, im.width - 20, 60):
        d.line((x, 20, x + 34, 20), fill=(255, 255, 255), width=6); d.line((x, im.height - 20, x + 34, im.height - 20), fill=(255, 255, 255), width=6)
    for y in range(20, im.height - 20, 60):
        d.line((20, y, 20, y + 34), fill=(255, 255, 255), width=6); d.line((im.width - 20, y, im.width - 20, y + 34), fill=(255, 255, 255), width=6)
    im.save(dst)


def two(o1, o2):
    return lambda f, tt, s, c: o2(o1(f, tt, s, c), tt, s, c)


def build():
    blueprint("shots/s3_topic_mars.png", "shots/s4_topic_mars_plan.png")
    if not os.path.exists("shots/s5_reyhan.png"):
        import build_scene5; build_scene5.earth_on_monitor()
    import build_scene2  # Scene 2 yuz setleri (wide2, reyhan2, alex2, maya2, noah2)
    b.FACES.update({
        "hayes": ("shots/s4_hayes.png", "shots/s4_hayes_closed.png", auto_box("shots/s4_hayes.png", "shots/s4_hayes_closed.png"), 3),
        "reyhan5": ("shots/s5_reyhan.png", "shots/s5_reyhan_closed.png", (560, 150, 800, 420)),
    })
    print("hayes kutusu", b.FACES["hayes"][2])
    audio = {}
    for f, ids in CLIPS.items():
        parts = align_split("audio/" + f, [TXT[i] for i in ids], EXTRA)
        for i, x in zip(ids, parts): audio[i] = x
    who = {k: w for k, w, _ in S}
    for k in audio:
        if who[k] in ("Hayes", "Station"): audio[k] = radio(audio[k])
    src = {i: f for f, ids in CLIPS.items() for i in ids}
    T = {k: (w, t, f"{src[k]} ({CLIPS[src[k]].index(k) + 1}. bölüm)" if len(CLIPS[src[k]]) > 1 else src[k]) for k, w, t in S}

    EL = (0.0, 0.04, 0.62, 0.66); KIDS = (0.45, 0.16, 1.0, 0.71)
    HA, HB = FULL, (0.04, 0.0, 0.86, 0.82)
    def hayes(lid, note, ov=None, flip=False):
        return dict(img="shots/s4_hayes.png", lid=lid, rule="line", anim="hayes", c0=HB if flip else HA, c1=HA if flip else HB, overlay=ov, note=note)
    def kid(name, lid, note, c1, ov=None, img=None):
        return dict(img=img or f"shots/s2_{name}.png", lid=lid, rule="line", anim=(name + "5") if img else (name + "2"), c0=FULL, c1=c1, overlay=ov, note=note)
    R1, A1, M1, N1 = (0.06, 0.0, 0.96, 0.9), (0.08, 0.02, 0.98, 0.92), (0.0, 0.05, 0.9, 0.95), (0.10, 0.06, 0.98, 0.94)
    PLAN = [
        dict(img="(grafik)", rule=2.4, anim="card", note="SAHNE GEÇİŞ KARESİ: “Scene 4 – The Big Connection” (seslendirilmez)."),
        dict(img="backgrounds/radio_room_wide.png", rule=1.8, c0=FULL, c1=(0.03, 0.03, 0.97, 0.97), note="Bağlantı günü, okulun radyo/stüdyo alanı. Bağlantı sesi duyulur, hafif radyo uğultusu başlar."),
        dict(img="(grafik: bağlantı göstergesi)", lid="1", rule="line", anim=link_screen, note="Bağlantı göstergesi etkinleşir; Station’ın anonsu radyodan duyulur, herkes dinler."),
        dict(img="backgrounds/radio_room_wide.png", lid="2", rule="line", anim="wide2", c0=FULL, c1=EL, note="Anons bittikten sonra Ms Ellis mikrofona yanıt verir."),
        dict(img="shots/iss.png", lid="3", rule=("f", 0.30), c0=FULL, c1=(0.08, 0.08, 0.92, 0.92), note="ISS dışarıdan görünür; Hayes’in sesi radyodan gelir."),
        dict(img="shots/s4_hayes.png", lid="3", rule="tail", anim="hayes", c0=HA, c1=HB, note="Commander Hayes ISS içinde, görev tulumuyla (ağız hareketi + göz kırpma)."),
        dict(img="backgrounds/radio_room_wide.png", lid="4", rule="line", anim="wide2", c0=EL, c1=(0.0, 0.0, 0.8, 0.8), note="Ms Ellis hazır olduklarını söyler ve Reyhan’a işaret eder."),
        kid("reyhan", "5", "Reyhan mikrofona yaklaşır ve selam verir.", R1),
        hayes("6", "Hayes ilk soruyu ister.", flip=True),
        kid("reyhan", "7", "Reyhan ilk soruyu sorar.", (0.10, 0.02, 0.94, 0.86)),
        hayes("8", "Hayes yanıtlar; önce oksijen izleme sistemi, ardından radyasyon kalkanı şeması gösterilir.", two(inset("shots/s3_topic_oxygen.png", 0.03, 0.50, "tr"), inset("shots/s3_topic_radiation.png", 0.52, 0.99, "tr"))),
        kid("reyhan", "9", "Reyhan su sorusunu sorar.", R1),
        hayes("10", "Hayes yanıtlar; uzayda süzülen su damlacıkları destek görüntüsü olarak gösterilir.", inset("shots/s4_topic_water.png", 0.10, 0.99, "tr"), flip=True),
        kid("alex", "11", "Alex soru sırasını alır.", A1),
        hayes("12", "Hayes şaka yapar; havada süzülen kurabiye, kırıntı ve kalemler gösterilir. Son sözünde gülümser.", inset("shots/s4_topic_floating.png", 0.18, 0.80, "tr")),
        dict(img="backgrounds/radio_room_wide.png", rule=1.1, c0=KIDS, note="Öğrencilerin kısa, doğal tepkisi (gülümserler)."),
        kid("reyhan", "13", "Reyhan gülerek karşılık verir.", R1),
        kid("maya", "14", "Maya soru sırasını alır; Mars kolonisi bir gelecek taslağı olarak görünür.", M1, inset("shots/s4_topic_mars_plan.png", 0.22, 0.99, "tr")),
        hayes("15", "Hayes yanıtlar.", flip=True),
        kid("noah", "16", "Noah, Hayes’in yanıtı bittikten sonra mikrofona yönelir; gezegen gözlem görseli gösterilir.", N1, inset("shots/s4_topic_planet.png", 0.28, 0.99, "tl")),
        hayes("17", "Hayes yanıtlar; gezegen arama görseli sürer.", inset("shots/s4_topic_planet.png", 0.14, 0.64, "tr")),
        kid("noah", "18", "Noah düşüncesini paylaşır.", (0.06, 0.04, 0.96, 0.94)),
        kid("alex", "19", "Alex konuşur; geri dönüşüm, su ve enerji tasarrufu görseli gösterilir.", A1, inset("shots/s4_topic_care.png", 0.10, 0.62, "tr")),
        kid("reyhan", "20", "Dünya’nın uzaydan görünüşü okuldaki ekrana gelir; Reyhan bu görüntüye bakarak konuşur.", R1, img="shots/s5_reyhan.png"),
        hayes("21", "Hayes yanıtlar; Dünya görüntüsü korunur.", inset("shots/s3_topic_earth.png", 0.04, 0.99, "tr"), flip=True),
        kid("reyhan", "22", "Reyhan son sorusunu sorar; ekranda Dünya görüntüsü durur.", (0.10, 0.02, 0.94, 0.86), img="shots/s5_reyhan.png"),
        hayes("23", "Hayes ailesinden söz eder; ton yumuşar."),
        dict(img="shots/s4_hayes_window.png", rule=1.8, c0=FULL, c1=(0.1, 0.05, 1.0, 0.95), note="Kısa sessizlik: Hayes pencereden Dünya’ya bakar."),
        dict(img="shots/s4_listening.png", rule=1.8, c0=FULL, c1=(0.04, 0.04, 0.96, 0.96), note="Kısa sessizlik: öğrencilerin dikkatle dinleyen yüzleri."),
        dict(img="backgrounds/radio_room_wide.png", lid="24", rule="line", anim="wide2", c0=EL, c1=(0.0, 0.03, 0.66, 0.69), note="Ms Ellis teşekkür eder."),
        dict(img="(grafik: bağlantı göstergesi)", lid="25", rule="line", anim=link_screen, steady=True, note="Station’ın kapanış anonsu; bağlantı hemen kesilmez, gösterge açık kalır."),
        kid("alex", "26", "Alex sözünü mevcut bağlantı sürerken tamamlar.", A1),
        dict(img="shots/s4_listening.png", rule=1.8, c0=(0.04, 0.04, 0.96, 0.96), c1=FULL, note="Kısa geçiş; sinyal ve teknik uğultu kesilir, kararma."),
    ]
    rng = np.random.default_rng(8)

    def hum(total):  # hafif radyo ugultusu: baglanti basindan sahne sonuna, sonda kesilir
        n = int((total - 2.6) * SR)
        x = signal.sosfilt(signal.butter(2, [200, 2500], btype="band", fs=SR, output="sos"), rng.standard_normal(n))
        x = 0.006 * x / np.abs(x).std()
        fi, fo = int(0.6 * SR), int(1.5 * SR); x[:fi] *= np.linspace(0, 1, fi); x[-fo:] *= np.linspace(1, 0, fo) ** 2
        x[-int(0.5 * SR):] = 0
        return 2.6, x.astype(np.float32)
    t = np.arange(int(0.45 * SR)) / SR
    beep = (0.09 * np.sin(2 * np.pi * 1000 * t) * ((t < 0.12) | ((t > 0.2) & (t < 0.32)))).astype(np.float32)
    render(4, "The Big Connection", T, audio, PLAN, sfx=[hum, (2.9, beep)],
           md_extra="\nMekânlar: okulun radyo/stüdyo alanı (`backgrounds/radio_room_wide.png`) ve ISS’in içi (`shots/s4_hayes.png`). Destek görselleri: `shots/s4_topic_*.png`, `shots/s3_topic_*.png`.\n",
           rules_md="""- "Scene 4 – The Big Connection" ayrı bir geçiş karesinde gösterilir, seslendirilmez.
- Station ve Hayes radyo üzerinden duyulur (hafif telsiz rengi); biri kadın, biri erkek sestir.
- Hayes ISS içinde görev tulumuyla gösterilir; radyo bağlantısı görüntülü görüşme gibi sunulmaz (Hayes okuldaki ekranda görünmez).
- Oksijen sistemi ve radyasyon kalkanı görselleri Hayes’in ilgili cümleleriyle eşleşir.
- Su damlacıkları, süzülen kalem ve kırıntılar destek görüntüsü olarak gösterilir.
- Mars kolonisi bir gelecek taslağı (mavi plan çizimi) olarak görünür.
- Reyhan Dünya görüntüsüne bakarak konuşur; sonraki yanıtta aynı görüntü korunur.
- Hayes’in ailesinden söz etmesinin ardından kısa bir sessizlik vardır.
- Station’ın kapanış anonsunda bağlantı kesilmez; Alex sözünü bitirir, sonra uğultu kesilir.
""")


if __name__ == "__main__":
    build()
