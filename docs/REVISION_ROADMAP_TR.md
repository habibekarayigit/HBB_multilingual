# Editör kararı sonrası revizyon yol haritası

## 1. Editörün eleştirisi ne anlama geliyor?

Bu karar, yalnızca yazım veya ek bir baseline eksikliği değildir. Editörün temel itirazı **katkı türüne** yöneliktir:

- Çalışmada kullanılan veri kümeleri ve modeller literatürden alınmış.
- Yenilik esas olarak “aglütinatif dillere uygulanmış olması” ile sınırlı kalmış.
- Morfolojik tipoloji iddiasını doğrudan sınayan bir model, kontrollü müdahale veya ayrıntılı dilbilimsel hata analizi bulunmuyor.
- Bu nedenle çalışma, *Language Resources and Evaluation* açısından yeni bir kaynak, değerlendirme altyapısı veya yöntem sunmayan artımlı bir karşılaştırma olarak görülmüş.

Bu bir “minor revision” değildir. Mevcut metni birkaç paragraf ve birkaç yeni baseline ile aynı dergiye geri göndermek yeterli olmaz. En doğru yaklaşım, önce deneysel kayıtları düzeltmek, sonra çalışmayı ya **yeni bir dil kaynağı/kontrollü benchmark** ya da **morfolojiye özgü bir yöntem ve mekanizma analizi** etrafında yeniden kurmaktır.

## 2. Yeniden gönderimden önce çözülmesi gereken veri ve kod sorunları

Depodaki `ATC (2).xlsx`, `RTP_LX_FI (1).xlsx`, `Cross_lingual.ipynb` ve sonuç dosyaları incelendi. Aşağıdaki sorunlar, editör eleştirisinden bağımsız olarak, yeni bir gönderimden önce çözülmelidir.

### 2.1 ATC sayıları makaledeki sayılarla uyuşmuyor

Sağlanan çalışma kitabında:

- 30.358 satır,
- 10.532 pozitif,
- 19.826 negatif örnek vardır.

Makalede ise 30.354 / 10.528 / 19.826 raporlanmıştır. Ayrıca çalışma kitabında:

- 992 tam yinelenen satır,
- muhafazakâr Unicode ve boşluk normalizasyonundan sonra 1.071 yinelenen metin,
- dört çelişkili metin grubu ve bu gruplarda 21 satır,
- bir boş metin bulunmaktadır.

Rastgele satır düzeyinde bölme yapılırsa aynı yorumun kopyaları train ve test kümelerine düşebilir ve monolingual sonuçları yapay biçimde yükseltebilir. Önce normalize edilmiş metne göre gruplama/deduplication, sonra split yapılmalıdır.

### 2.2 Finlandiya etiket sayıları önemli ölçüde yanlış raporlanmış

`CompletionAnnotations.Toxicity` sütunu tamsayıdır. Dolayısıyla kodda kullanılan `>= 3.5` koşulu fiilen `>= 4` ile aynıdır. Sağlanan 1.099 satırda bu kural:

- 392 pozitif (%35,7),
- 707 negatif (%64,3)

üretmektedir. Makaledeki 157 pozitif (%14,3) sayısı tam veri kümesine ait değildir. 157, 1.099 satırın %40'lık test bölümündeki yaklaşık pozitif sayıdır. Kod yalnızca 440 satırlık test bölümünü oluştururken, makale tablosu bu test bölümündeki 157 pozitif örneği yanlışlıkla tüm veri kümesinin toplam pozitif sayısı gibi kullanmış görünmektedir.

Bu nedenle en az Tablo 3 ve Tablo 4 yeniden hesaplanmalıdır. `>=4` eşiği kullanılacaksa beklenen 560/99/440 bölmesinin yaklaşık sınıf sayıları şöyledir:

| Split | Negatif | Pozitif | Toplam |
|---|---:|---:|---:|
| Train | 360 | 200 | 560 |
| Dev | 64 | 35 | 99 |
| Test | 283 | 157 | 440 |

Eşik “en dengeli dağılımı verdiği için” test verisine bakılarak seçilmemelidir. Eşik veri kümesinin özgün tanımına veya önceden belirlenmiş bir kurala dayanmalı; komşu eşiklerle duyarlılık analizi ayrıca raporlanmalıdır.

### 2.3 Makaledeki deneylerin çoğu sağlanan notebook'tan yeniden üretilemiyor

Depodaki sonuç çalışma kitabı yalnızca 200 örneklik baseline karşılaştırmalarını içeriyor. Sağlanan notebook içinde şu nihai deneylerin tam ve temiz-kurulum uygulaması görünmüyor:

- üç seed ile dört encoder,
- XLM-R-large ve Glot500 ana deneyleri,
- Gemma 3 sonuçları,
- tüm few-shot koşulları,
- transfer matrisi,
- joint-training koşulları,
- makaledeki morfolojik korelasyon tablosu.

Notebook ayrıca önceki hücrelerin bellekte bıraktığı değişkenlere bağımlı. Örneğin `df_tr_test` V4 bloğunda tanımlanmadan kullanılıyor; ilk frozen-encoder deneyi encoder'ı dondurduktan sonra yeni ve tamamen eğitilebilir bir model oluşturuyor; bir LaBSE sonuç listesi `return` satırından sonra yanlış girintilendiği için erişilemez; `predict_fast` tüm test kümesini tek GPU batch'inde çalıştırıyor. Notebook'taki alakasız uyku/kahve ve ekip-verimliliği hücreleri de yeniden üretilebilir araştırma paketi izlenimini zayıflatıyor.

**Sonuç:** Mevcut makale tabloları, her satırı temiz bir çalıştırmadan üretilen makine-okunur sonuç ve örnek-düzeyi tahmin dosyasına bağlanmadan yeniden kullanılmamalıdır.

## 3. Önerilen ana yön: morfoloji-kontrollü benchmark + mekanizma analizi

LRE kapsamına en güçlü uyum sağlayacak seçenek, mevcut “birkaç modelin birkaç dilde karşılaştırılması” çalışmasını aşağıdaki somut katkıya dönüştürmektir:

> **Morfolojik olarak açıklamalı ve kontrollü, dillerarası saldırgan dil değerlendirme kaynağı; sabit splitler, minimal çiftler, morfolojik müdahaleler ve örnek-düzeyi hata etiketleri.**

Bu yaklaşım iki sorunu birlikte çözer:

1. Dergiye yeni bir **language resource/evaluation protocol** sunar.
2. “Aglütinasyon neden yardımcı olabilir?” sorusunu yalnızca dil ortalamalarıyla değil, doğrudan müdahalelerle sınar.

### 3.1 Önce görev adını düzeltin

ATC “abusive”, BEEP! “hate/aggressive”, TweetEval “offensive”, RTP-LX “toxicity” ve Macarca veri kümesi “toxic” etiketleri aynı kavram değildir. Tümünü “hate speech” diye adlandırmak yapı geçerliliği sorunu yaratır.

İki savunulabilir seçenek vardır:

- Ana görevi **binary abusive/offensive language detection** olarak yeniden adlandırmak ve her veri kümesinin özgün etiket haritasını açıkça vermek.
- Yalnız gerçekten aynı hedef tanımını karşılayan veri kümelerini ana analize almak; diğerlerini domain/construct-robustness analizi olarak ayırmak.

### 3.2 Tek İngilizce kontrol yeterli değildir

Dört aglütinatif hedefe karşı yalnızca bir füzyonel dil kullanmak, tipolojiyi veri kümesi, domain, kaynak büyüklüğü ve pretraining kapsamından ayıramaz. Dört hedef dil üzerinden Spearman korelasyonu tipolojik bir genelleme için yeterli örneklem değildir.

Tercih sırasına göre:

1. Aynı veya paralel değerlendirme kaynağında en az üç ek aglütinatif olmayan dil ekleyin.
2. Latin ve Latin olmayan yazı sistemlerini her iki tipoloji grubunda dengeleyin.
3. Bu mümkün değilse çalışmayı “tipolojik etki kanıtı” yerine açıkça **keşifsel beş-dil vaka çalışması** olarak çerçeveleyin ve dil-düzeyi korelasyon/p-değeri iddialarını çıkarın.

### 3.3 Kontrollü minimal çift kaynağı oluşturun

Her dil için ana dili konuşan uzmanlarla aynı olguları kapsayan örnekler oluşturun veya doğrulayın. Önerilen olgu sınıfları:

- çekimsel ekler ve kişi/iyelik işaretleme,
- yapım ekleriyle türetilen saldırgan ifadeler,
- olumsuzluk,
- alıntılanan/raporlanan saldırgan ifade,
- hedefe doğrudan yönelme,
- birleşik sözcük ve yazım varyantları,
- ekler arasına karakter koyarak gizleme,
- argo/küfür olup hedef göstermeyen kullanım,
- geri sahiplenilmiş veya ironi içeren kullanım.

Her temel içerik için mümkün olduğunda bir saldırgan ve bir saldırgan olmayan minimal çift üretin. Etiketleyiciler metnin sınıfına ek olarak `phenomenon`, `target`, `negation`, `quotation`, `morphological_process` ve `confidence` alanlarını doldursun. En az iki bağımsız değerlendirici ve uyuşmazlık çözümü kullanın; Cohen's kappa veya Krippendorff's alpha raporlayın.

RTP-LX dosyasında eşik üstündeki 392 `Completion` satırının her birinde bir `BenignCompletion` karşılığı vardır ve bunlar 392 çift/784 metinlik bir **eşleştirilmiş tanı testi** sağlayabilir. Ancak yaptığımız tokenizasyon denetiminde toksik metinlerin medyanı 1.205, benign karşılıkların medyanı 194 karakterdir; toksiklerin %99'u 128 tokenda kesilirken benignlerin hiçbiri kesilmez. Bu nedenle mevcut çiftler morfoloji-kontrollü minimal çift veya temiz counterfactual kanıt sayılmamalıdır. Uzunluk/içerik eşleştirmesi yapılmalı veya yeni minimal çiftler uzmanlarca üretilmelidir. Her durumda aynı çifte ait iki metin farklı splitlere gitmemelidir.

## 4. Morfolojiyi doğrudan sınayan deney tasarımı

Sadece tokenizer fragmentation oranı morfolojik mekanizmayı göstermez. Aşağıdaki koşullar aynı örneklerde karşılaştırılmalıdır:

1. **Surface baseline:** özgün metin.
2. **Lemma ablation:** yüzey sözcükleri evrensel bağımlılık çözümleyicisinden gelen lemmalarla değiştirilir; çekimsel bilgi büyük ölçüde kaldırılır.
3. **Lemma + UD morphology:** lemma yanında UPOS ve `Case`, `Number`, `Person`, `Tense`, `Mood`, `Polarity` gibi UD FEATS etiketleri verilir.
4. **Randomized-tag placebo:** morfolojik etiketler aynı frekans dağılımı korunarak örnekler arasında karıştırılır. Kazanç gerçek morfolojik bilgiden geliyorsa bu koşulda kaybolmalıdır.
5. **Korean romanization control:** yalnızca script etkisini inceleyen ikincil koşul. Romanizasyon bir morfoloji yöntemi gibi sunulmamalıdır.

Stanza/UDPipe gibi aynı Universal Dependencies çerçevesini kullanan çözümleyiciler diller arası karşılaştırılabilirlik sağlar. Ancak sosyal medya metnindeki çözümleyici hataları, her dilden rastgele en az 100 örnek üzerinde manuel olarak denetlenmelidir.

### 4.1 Somut model önerisi: çift-görünümlü morfolojik tutarlılık

Yeni model katkısı istenirse, ortak encoder ile iki görünüm kullanılabilir:

- `x_surface`: özgün metin,
- `x_morph`: lemma + evrensel morfolojik etiket dizisi.

Aynı XLM-R encoder iki görünümü işler. Eğitim kaybı:

`L = CE(y, p_surface) + CE(y, p_morph) + lambda * JS(p_surface || p_morph)`

olabilir. Testte iki görünümün olasılıkları ortalanır. Gerekli ablationlar:

- yalnız surface,
- yalnız lemma,
- yalnız morphology view,
- iki görünüm fakat consistency loss yok,
- iki görünüm + consistency loss,
- rastgeleleştirilmiş morfolojik etiketler.

Bu modelin gerçekten yeni olduğu iddia edilmeden önce morphology-aware multilingual classification ve consistency-training literatürü sistematik olarak taranmalıdır. Eğer yöntemsel yenilik yeterince güçlü değilse, yöntem “araştırma aracı” olarak sunulmalı ve ana katkı kontrollü benchmark/linguistik analiz olmalıdır.

## 5. Nitel hata analizi nasıl yapılmalı?

Her dil ve ana model için FP, FN, TP ve TN hücrelerinden körlenmiş ve dengeli örnek seçin. En az iki dil uzmanı aşağıdaki şemayla etiketlesin:

- `inflectional_suffix`
- `derivational_suffix`
- `negation`
- `quotation_or_reported_speech`
- `target_is_explicit`
- `profanity_without_abuse`
- `obfuscation`
- `compound_or_spacing`
- `irony_or_reclaimed_use`
- `annotation_ambiguous`
- `domain_or_cultural_reference`

Sadece birkaç örnek vermek yerine, her olgu için model hata oranı ve güven aralığı raporlayın. Örneğin “olumsuzluk içeren minimal çiftlerde FN oranı” veya “yapım eki içeren örneklerde morphology-aware modelin baseline'a göre hata azalması” doğrudan dilbilimsel bir sonuç üretir.

## 6. İstatistiksel protokol

- Her öğrenilen model/koşul için aynı üç veya tercihen beş seed kullanılmalı.
- Ana metrik Macro-F1; abusive-class F1, precision/recall, MCC ve calibration ikincil metrik olabilir.
- Aynı test örnekleri üzerindeki model farkları için **paired bootstrap** veya yaklaşık randomization testi kullanılmalı.
- Minimal çift kaynağında bootstrap `pair_id` düzeyinde kümelenmeli.
- FP/FN karşılaştırmaları için McNemar testi uygundur.
- Çoklu karşılaştırmalar Holm düzeltmesiyle kontrol edilmelidir.
- Etki büyüklüğü `Delta Macro-F1` ve %95 güven aralığıyla verilmelidir.
- Dört hedef dil üzerinde p-değerli tipoloji korelasyonu raporlanmamalıdır. Dil-düzeyi regresyon ancak yeterli sayıda ve dengeli dil eklendiğinde anlamlıdır.

## 7. Deneylerin yeniden üretilebilir biçimde düzenlenmesi

Her çalıştırma aşağıdaki dosyaları üretmelidir:

- `run_config.json`: model, veri hash'i, split manifest hash'i, seed, hiperparametreler, paket sürümleri;
- `metrics.json`: tüm metrikler;
- `predictions.parquet/csv`: örnek ID, gold label, prediction, sınıf olasılıkları;
- `confusion_matrix.csv`;
- eğitim logu ve seçilen checkpoint bilgisi.

Makale tabloları bu dosyalardan otomatik oluşturulmalıdır. Full test ve 200-sample baseline sonuçları aynı tabloda karıştırılmamalıdır. Her tablonun başlığında `N`, split manifesti, seed sayısı ve full/subset durumu yazmalıdır.

Kaynakların lisansları ham metni yeniden dağıtmaya izin vermiyorsa yalnız loader, dosya hash'i, satır ID'si/split manifesti ve veri kartı yayımlanmalıdır.

## 8. Makale nasıl yeniden çerçevelenmeli?

### Güçlü LRE sürümü için olası başlık

**A Morphology-Controlled Multilingual Benchmark for Cross-Lingual Abusive Language Detection**

Bu sürümün ana katkıları şöyle olmalıdır:

1. Sabit, leakage-safe split ve ortak etiket eşleme protokolü.
2. Beş veya daha fazla dilde morfolojik olgularla açıklanmış kontrollü minimal çift kaynağı.
3. Lemma/morfolojik-etiket müdahaleleriyle mekanizma analizi.
4. Tam örnek-düzeyi tahminler, veri kartları ve tekrar üretim kodu.

### Yeni kaynak oluşturulamayacaksa

Makale “morfolojik tipolojinin kanıtı” olarak değil, sınırlı kapsamlı keşifsel karşılaştırma olarak yeniden yazılmalı; nedensel ve genelleyici iddialar azaltılmalı; mevcut dil-düzeyi korelasyon tablosu çıkarılmalı ve uygulamalı/ampirik karşılaştırmaları kabul eden bir derginin güncel kapsamı kontrol edilmelidir.

## 9. Uygulanabilir çalışma sırası

1. **Veri dondurma ve audit:** dosya hash'leri, deduplication, etiketi doğrulama, sabit splitler.
2. **Temiz baseline:** tek script/notebook, batch tahmin, üç seed, tam test, örnek-düzeyi çıktılar.
3. **Kavramsal hizalama:** “hate/offensive/toxic” etiket haritası ve görev adının düzeltilmesi.
4. **Morfolojik müdahaleler:** surface, lemma, lemma+FEATS, randomized FEATS.
5. **Kontrollü kaynak:** minimal çiftler ve iki uzmanla anotasyon.
6. **Nitel analiz:** hata örneklemesi ve olgu bazlı sonuçlar.
7. **İstatistik:** paired testler, güven aralıkları, çoklu karşılaştırma düzeltmesi.
8. **Makale yazımı:** sonuç dosyalarından otomatik tablo üretimi; iddiaların yalnız elde edilen kanıtla sınırlandırılması.

## 10. Bu depoya eklenen ilk düzeltme

`scripts/audit_and_build_splits.py`:

- iki yerel Excel dosyasını denetler,
- ATC tekrarlarını/çelişkilerini sayar,
- metni paylaşmadan hash tabanlı, leakage-safe 8.000/1.000/1.000 ATC manifesti üretir,
- Finlandiya eşiklerini ve doğru sınıf sayılarını raporlar,
- 560/99/440 Finlandiya split manifesti üretir,
- 392 toksik/benign eşleşmenin manifestini oluşturur.

Çalıştırma:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-audit.txt
.venv/bin/python scripts/audit_and_build_splits.py
```

Üretilen rapor: `artifacts/audit/data_audit.md`.

`scripts/build_ud_morph_views.py`, standartlaştırılmış `example_id, language, text, label, split` girdisinden `lemma_text`, `morph_text` ve frekans dağılımını koruyan `randomized_morph_text` placebo görünümü üretir. Bu dosya morfolojik müdahale deneylerinin veri hazırlama adımıdır; otomatik çözümleyicinin sosyal medya doğruluğu ayrıca manuel denetlenmelidir.

`scripts/sample_linguistic_errors.py`, örnek-düzeyi tahminlerden her dil için dengeli TP/TN/FP/FN örnekleri seçer ve model sonucunu anotatörden gizleyen iki ayrı dosya üretir.
