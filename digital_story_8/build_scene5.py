#!/usr/bin/env python3
"""Digital Story 8 - Scene 5 "After the Signal".  Calistirma: python3 build_scene5.py"""
import os
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
import build_scene1 as b
from story_engine import render, auto_box, FULL

os.chdir(os.path.dirname(os.path.abspath(__file__)))

T = {
    "1": ("Ms Ellis", "You did it, everyone. You spoke to someone orbiting above our planet.", "s5_01_ellis.wav"),
    "2": ("Maya", "It feels strange… Seeing the Earth from up there makes it feel smaller, not bigger.", "s5_02_maya.wav"),
    "3": ("Noah", "I keep thinking about what the Commander said: “If we don’t cooperate, we won’t survive long in space.”", "s5_03_noah.wav"),
    "4": ("Alex", "That's true for Earth, too. If we don’t work together, nothing works.", "s5_04_alex.wav"),
    "5": ("Reyhan", "I used to dream about living on another planet. Now I just want to help protect this one.", "s5_05_reyhan.wav"),
    "6": ("Ms Ellis", "That’s the spirit. Science is not only about discovery; it’s about care. All right, team! Let’s write down what today has taught us.", "s5_06_ellis.wav"),
    "7": ("Narration", "If curiosity starts small, responsibility potentially does, too.", "s5_07_narration.wav"),
}


def earth_on_monitor():
    """Scene 2 yakin planlarindaki duvar ekranina (dalga formu) Dunya goruntusu yerlestirir -> s5_*.png."""
    x0, y0, x1, y1 = 933, 78, 1316, 308
    pic = Image.open("shots/s3_topic_earth.png").convert("RGB").crop((238, 114, 1138, 654)).resize((x1 - x0, y1 - y0), Image.LANCZOS)
    for name in ("reyhan", "alex", "maya", "noah"):
        A = Image.open(f"shots/s2_{name}.png").convert("RGB")
        arr = np.asarray(A).astype(int); r, g, bl = arr[..., 0], arr[..., 1], arr[..., 2]
        scr = ((g > r + 12) & (bl > r + 5) & (g < 140) & (r < 95)) | ((g > 150) & (g > r + 40) & (g > bl + 20))   # ekran zemini + yesil cizgi
        box = np.zeros_like(scr); box[y0:y1, x0:x1] = True
        m = ndimage.binary_fill_holes(ndimage.binary_closing(scr & box, iterations=3)) & box
        mask = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
        for suf in ("", "_closed"):
            im = Image.open(f"shots/s2_{name}{suf}.png").convert("RGB").resize(A.size, Image.LANCZOS)
            im.paste(pic, (x0, y0), mask.crop((x0, y0, x1, y1)))
            im.save(f"shots/s5_{name}{suf}.png")


def build():
    earth_on_monitor()
    b.FACES.update({
        "ellis5": ("shots/s5_wide_talk.png", "shots/s5_wide_closed.png", auto_box("shots/s5_wide_talk.png", "shots/s5_wide_closed.png")),
        "reyhan5": ("shots/s5_reyhan.png", "shots/s5_reyhan_closed.png", (560, 150, 800, 420)),
        "alex5": ("shots/s5_alex.png", "shots/s5_alex_closed.png", (400, 170, 660, 430)),
        "maya5": ("shots/s5_maya.png", "shots/s5_maya_closed.png", (280, 180, 540, 450)),
        "noah5": ("shots/s5_noah.png", "shots/s5_noah_closed.png", (640, 180, 950, 450), 3),
    })
    print("ellis5 kutusu", b.FACES["ellis5"][2])
    audio = {k: b.load_wav("audio/" + v[2]) for k, v in T.items()}
    EL = (0.0, 0.06, 0.60, 0.66)
    KIDS = (0.42, 0.10, 1.0, 0.68)
    PLAN = [
        dict(img="(grafik)", rule=2.4, anim="card", note="SAHNE GEÇİŞ KARESİ: “Scene 5 – After the Signal” (seslendirilmez)."),
        dict(img="shots/s5_wide_after.png", rule=2.4, c0=KIDS, c1=FULL, note="Bağlantı sonrası aynı okul alanı; teknik uğultu yok. Öğrenciler kulaklıklarını çıkarır, ekranda Dünya görüntüsü kalır."),
        dict(img="shots/s5_wide_talk.png", lid="1", rule="line", anim="ellis5", c0=FULL, c1=EL, note="Ms Ellis gururlu ve sıcak konuşur (ağız hareketi + göz kırpma)."),
        dict(img="shots/s5_maya.png", lid="2", rule="line", anim="maya5", c0=FULL, c1=(0.0, 0.04, 0.92, 0.96), note="Maya ekrandaki Dünya görüntüsüne bakarak konuşur."),
        dict(img="shots/s5_noah.png", lid="3", rule="line", anim="noah5", c0=(0.08, 0.06, 0.98, 0.96), c1=(0.14, 0.10, 0.98, 0.94), note="Noah Komutan’ın sözünü hatırlatır."),
        dict(img="shots/s5_alex.png", lid="4", rule="line", anim="alex5", c0=FULL, c1=(0.08, 0.02, 0.98, 0.92), note="Alex ekrandaki Dünya’yı işaret ederek konuşur."),
        dict(img="shots/s5_reyhan.png", lid="5", rule="line", anim="reyhan5", c0=FULL, c1=(0.06, 0.0, 0.96, 0.9), note="Reyhan düşüncesini paylaşır."),
        dict(img="shots/s5_wide_talk.png", lid="6", rule="line", anim="ellis5", c0=EL, c1=FULL, note="Ms Ellis toparlar ve öğrencileri yazmaya yönlendirir; kamera odanın geneline açılır."),
        dict(img="shots/s5_writing.png", rule=2.2, c0=FULL, c1=(0.04, 0.03, 1.0, 0.99), note="Öğrenciler defterlerini açıp yazmaya başlar; kısa, doğal bir sessizlik."),
        dict(img="shots/s5_writing.png", lid="7", rule="line", c0=(0.04, 0.03, 1.0, 0.99), c1=(0.34, 0.22, 1.0, 0.88), note="Anlatıcı dış sestir, ekranda görünmez; öğrenciler yazmayı sürdürür."),
        dict(img="shots/s5_writing.png", rule=2.4, c0=(0.34, 0.22, 1.0, 0.88), c1=FULL, note="Sakin genel görünüm kısa bir an korunur ve sahne kapanır (kararma)."),
    ]
    render(5, "After the Signal", T, audio, PLAN,
           md_extra="\nMekân: Scene 2 ile aynı radyo/stüdyo alanı; duvar ekranında Dünya görüntüsü vardır (`shots/s5_wide_after.png`).\n",
           rules_md="""- "Scene 5 – After the Signal" ayrı bir geçiş karesinde gösterilir, seslendirilmez.
- Teknik uğultu kullanılmaz; tempo sakindir.
- Öğrenciler kulaklıklarını çıkarır, ekranda Dünya görüntüsü kalır.
- Maya ekrandaki Dünya görüntüsüne bakar.
- Ms Ellis’in yönlendirmesinden sonra öğrenciler yazmaya başlar; kısa bir sessizlikten sonra anlatıcı girer.
- Anlatıcı dış sestir; sahneye yeni bir karakter olarak girmez.
- Anlatıcının sözü bitince genel görünüm kısa bir an korunur ve sahne kapanır.
""")


if __name__ == "__main__":
    build()
