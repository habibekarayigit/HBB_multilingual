#!/usr/bin/env python3
"""Digital Story 8 - Scene 3 "Planning the Big Question".  Calistirma: python3 build_scene3.py"""
import os
from PIL import Image, ImageDraw
import build_scene1 as b
from build_scene1 import W, H, GREEN, font, ease, fade_text, SR
from story_engine import render, inset, split_clip, FULL

os.chdir(os.path.dirname(os.path.abspath(__file__)))

T = {
    "1": ("Ms Ellis", "The final call schedule just came in. Next Friday you’ll speak live with the astronauts. Think about what you really want to know.", "s3_01_ellis.wav"),
    "2": ("Reyhan", "I want to ask some questions about their mission, like how long they’ve been up there and what they’re exploring.", "s3_02_reyhan.wav"),
    "3": ("Alex", "Maybe we can ask about oxygen levels or how they handle radiation.", "s3_03+09_alex.wav (1. bölüm)"),
    "4": ("Maya", "What about the view? They see the whole globe from up there; it must look so mysterious. And what’s happening elsewhere in space?", "s3_04_maya.wav"),
    "5": ("Noah", "I read that space navigation is much more sophisticated than aircraft navigation, so I wonder how they keep everything working all the time. Maybe we should ask that.", "s3_10+05_noah.wav (2. bölüm)"),
    "6": ("Ms Ellis", "Excellent ideas. You are all thinking like astronauts already.", "s3_06_ellis.wav"),
    "7": ("Reyhan", "If there’s time, can we ask about future projects, like new satellites or colonies on Mars, and what comes beyond?", "s3_07_reyhan.wav"),
    "8": ("Maya", "That’s a great one. Or something more practical, like how they interact during daily tasks, why personal privacy is important on a mission, and how they cope with zero-gravity.", "s3_08_maya.wav"),
    "9": ("Alex", "We should keep a database of all our questions so we don’t repeat anything.", "s3_03+09_alex.wav (2. bölüm)"),
    "10": ("Noah", "The more I think about it, the more I want to join a space mission myself.", "s3_10+05_noah.wav (1. bölüm)"),
    "11": ("Reyhan", "Me too. Ms Ellis, would you go to space if you could?", "s3_11_reyhan.wav"),
    "12": ("Ms Ellis", "Maybe one day, when interplanetary travel becomes as easy as booking a flight. Until then, let’s make sure our questions are clear and our microphones are working efficiently.", "s3_12_ellis.wav"),
}


def calendar(tt, seg, ctx):
    """Baglanti takvimi: bir sonraki cuma isaretli."""
    im = Image.new("RGBA", (W, H), (244, 239, 226, 255)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((120, 70, W - 120, H - 70), radius=30, fill=(255, 255, 255, 255), outline=(40, 60, 110, 255), width=6)
    d.rounded_rectangle((120, 70, W - 120, 170), radius=30, fill=(40, 60, 110, 255)); d.rectangle((120, 140, W - 120, 170), fill=(40, 60, 110, 255))
    fade_text(im, W // 2, 92, "CALL SCHEDULE", font(46), (255, 255, 255), 1)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]; x0, cw = 330, 110
    d = ImageDraw.Draw(im)
    for i, dn in enumerate(days):
        w = d.textlength(dn, font=font(30)); d.text((x0 + i * cw + cw / 2 - w / 2, 200), dn, font=font(30), fill=(40, 60, 110))
    for r, lab in enumerate(("This week", "Next week")):
        y = 270 + r * 150
        d.text((160, y + 38), lab, font=font(26, False), fill=(90, 90, 90))
        for i in range(7):
            d.rounded_rectangle((x0 + i * cw + 8, y, x0 + (i + 1) * cw - 8, y + 110), radius=14, fill=(236, 240, 248) if i < 5 else (246, 240, 236))
    cx, cy = x0 + 4 * cw + cw / 2, 270 + 150 + 55
    p = ease((tt - 0.5) / 0.6)
    if p > 0:
        d.rounded_rectangle((x0 + 4 * cw + 8, 420, x0 + 5 * cw - 8, 530), radius=14, fill=(46, 204, 113))
        d.arc((cx - 70, cy - 72, cx + 70, cy + 72), -90, -90 + 360 * p, fill=(220, 60, 50), width=8)
    fade_text(im, W // 2, 580, "Live contact with the ISS", font(36), (40, 60, 110), ease((tt - 1.0) / 0.5), dy=0)
    return im.convert("RGB")


ITEMS = [("their mission", "2"), ("oxygen levels", "3"), ("radiation", "3"), ("the view from space", "4"), ("space navigation", "5"),
         ("future projects", "7"), ("daily tasks", "8"), ("personal privacy", "8"), ("zero-gravity", "8")]


def question_list(tt, seg, ctx):
    """Soru listesi (ogrencilerin sozunu ettigi konular) - 'database' repliginde gosterilir."""
    im = Image.new("RGBA", (W, H), (30, 86, 62, 255)); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, H), outline=(150, 105, 60, 255), width=26)
    fade_text(im, W // 2, 60, "OUR QUESTIONS", font(54), (255, 255, 255), 1)
    d = ImageDraw.Draw(im); d.line((W // 2 - 240, 135, W // 2 + 240, 135), fill=(255, 255, 255, 200), width=4)
    for i, (txt, _) in enumerate(ITEMS):
        p = ease((tt - 0.15 - i * 0.22) / 0.3)
        if p <= 0: continue
        col, row = divmod(i, 5); x = 150 + col * 560; y = 185 + row * 96
        lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); dl = ImageDraw.Draw(lay); al = int(255 * p)
        dl.ellipse((x, y + 8, x + 40, y + 48), outline=(255, 255, 255, al), width=4)
        dl.line((x + 9, y + 28, x + 18, y + 38, x + 33, y + 16), fill=GREEN + (al,), width=6)
        dl.text((x + 62, y), txt, font=font(40, False), fill=(255, 255, 255, al))
        im.alpha_composite(lay)
    return im.convert("RGB")


def build():
    audio = {k: b.load_wav("audio/" + v[2]) for k, v in T.items() if "+" not in v[2]}
    audio["3"], audio["9"] = split_clip("audio/s3_03+09_alex.wav", (len(T["3"][1]), len(T["9"][1])))
    audio["10"], audio["5"] = split_clip("audio/s3_10+05_noah.wav", (len(T["10"][1]), len(T["5"][1])))
    for k in ("3", "9", "10", "5"): print("bolunen replik", k, round(len(audio[k]) / SR, 2), "sn")
    E0, E1 = (0.06, 0.13, 0.93, 1.0), (0.12, 0.17, 0.90, 0.95)
    WIDE_E = (0.0, 0.12, 0.62, 0.74)
    PLAN = [
        dict(img="(grafik)", rule=2.4, anim="card", note="SAHNE GEÇİŞ KARESİ: “Scene 3 – Planning the Big Question” (seslendirilmez)."),
        dict(img="shots/s3_ellis_tablet.png", rule=2.2, c0=FULL, c1=(0.0, 0.1, 0.66, 0.76), note="Aynı sınıf, şeref köşesi görünür. Ms Ellis gelen bağlantı takvimini tabletten kontrol eder."),
        dict(img="shots/ellis_medium.png", lid="1", rule=("f", 0.30), anim="ellis", c0=E0, c1=E1, note="Ms Ellis sınıfa döner ve haberi verir (ağız hareketi + göz kırpma)."),
        dict(img="(grafik: takvim)", lid="1", rule=("f", 0.36), anim=calendar, note="Bağlantı takvimi: bir sonraki cuma işaretlenir."),
        dict(img="shots/ellis_medium.png", lid="1", rule="tail", anim="ellis", c0=E1, c1=E0, note="Ms Ellis öğrencilerden düşünmelerini ister."),
        dict(img="shots/reyhan.png", lid="2", rule="line", anim="reyhan", c0=FULL, c1=(0.05, 0.03, 0.95, 0.93), overlay=inset("shots/s3_topic_mission.png", 0.22, 0.98, "tl"), note="Reyhan konuşur; köşe penceresinde görev görseli belirir."),
        dict(img="shots/alex.png", lid="3", rule="line", anim="alex", c0=FULL, c1=(0.0, 0.0, 0.9, 0.9), overlay=lambda f, tt, s, c: inset_rad(inset_oxy(f, tt, s, c), tt, s, c), note="Alex konuşur; önce oksijen sistemi, sonra radyasyondan korunma görseli belirir."),
        dict(img="shots/maya.png", lid="4", rule="line", anim="maya", c0=FULL, c1=(0.08, 0.02, 0.98, 0.92), overlay=inset("shots/s3_topic_earth.png", 0.18, 0.98, "tl"), note="Maya konuşur; Dünya görseli belirir."),
        dict(img="shots/noah.png", lid="5", rule="line", anim="noah", c0=(0.02, 0.05, 0.92, 0.95), c1=FULL, overlay=inset("shots/s3_topic_navigation.png", 0.06, 0.98, "tr", 380), note="Noah ekrandaki görev rotasını işaret ederek konuşur; önerisi listeye eklenir."),
        dict(img="shots/ellis_medium.png", lid="6", rule="line", anim="ellis", c0=E0, c1=E1, note="Ms Ellis öğrencileri över."),
        dict(img="shots/reyhan.png", lid="7", rule="line", anim="reyhan", c0=(0.05, 0.03, 0.95, 0.93), c1=FULL, overlay=inset("shots/s3_topic_mars.png", 0.30, 0.98, "tl"), note="Reyhan konuşur; yeni uydu ve Mars kolonisi görseli belirir."),
        dict(img="shots/maya.png", lid="8", rule="line", anim="maya", c0=(0.08, 0.02, 0.98, 0.92), c1=FULL, overlay=inset("shots/s3_topic_daily.png", 0.28, 0.98, "tl"), note="Maya konuşur; uzayda günlük yaşam görseli belirir."),
        dict(img="shots/alex.png", lid="9", rule=("f", 0.38), anim="alex", c0=(0.0, 0.0, 0.9, 0.9), c1=(0.03, 0.02, 0.91, 0.90), note="Alex soruları kaydetmeyi önerir."),
        dict(img="(grafik: soru listesi)", lid="9", rule="tail", anim=question_list, note="Soru listesi: konuşulan konular sırayla listede belirir."),
        dict(img="(grafik: soru listesi)", rule=1.2, anim=lambda tt, s, c: question_list(tt + 10, s, c), note="Soru listesi kısa süre ekranda kalır."),
        dict(img="shots/noah.png", lid="10", rule="line", anim="noah", c0=FULL, c1=(0.06, 0.06, 0.94, 0.94), note="Noah hevesle konuşur."),
        dict(img="shots/reyhan.png", lid="11", rule="line", anim="reyhan", c0=FULL, c1=(0.08, 0.04, 0.96, 0.92), note="Reyhan Ms Ellis’e sorar."),
        dict(img="shots/ellis_medium.png", lid="12", rule=("f", 0.50), anim="ellis", c0=E0, c1=E1, note="Ms Ellis gülümseyerek yanıtlar."),
        dict(img="backgrounds/classroom_wide.png", lid="12", rule="tail", anim="wide", c0=WIDE_E, c1=FULL, note="Ms Ellis konuşmayı sürdürür; kamera sınıfın geneline açılır."),
        dict(img="backgrounds/classroom_wide.png", rule=1.4, note="Sınıfın genel görünümü; kararma. Canlı bağlantı henüz başlamamıştır."),
    ]
    render(3, "Planning the Big Question", T, audio, PLAN,
           md_extra="\nMekân: Scene 1 ile aynı sınıf (`backgrounds/classroom_wide.png`). Konu görselleri: `shots/s3_topic_*.png`.\n",
           rules_md="""- "Scene 3 – Planning the Big Question" ayrı bir geçiş karesinde gösterilir, seslendirilmez.
- Aynı sınıf kullanılır; şeref köşesi görünür.
- Ms Ellis önce bağlantı takvimini kontrol eder; takvimde bir sonraki cuma işaretlidir.
- Konuşulan konuya göre görev, oksijen sistemi, radyasyondan korunma, Dünya, uzayda yön bulma ve Mars görselleri gösterilir.
- Soru listesi konuşma ilerledikçe oluşur.
- Canlı bağlantı bu sahnede başlamaz; astronot görünmez.
""")


inset_oxy = inset("shots/s3_topic_oxygen.png", 0.28, 0.66, "tr")
inset_rad = inset("shots/s3_topic_radiation.png", 0.66, 0.99, "tr")

if __name__ == "__main__":
    build()
