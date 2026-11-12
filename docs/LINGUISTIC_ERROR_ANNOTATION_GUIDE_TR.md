# Dilbilimsel hata anotasyonu kısa kılavuzu

## Amaç

Bu anotasyon modelin tahminini yeniden etiketlemek için değil, doğru ve yanlış tahminlerle ilişkili dilsel olguları belirlemek içindir. Anotatör model adı, tahmin, güven skoru ve FP/FN bilgisini görmemelidir.

## İş akışı

1. Üç tam seed'in olasılıklarını birleştirin; tek bir “iyi görünen” seed seçmeyin:

```bash
python scripts/ensemble_encoder_predictions.py
```

2. Önceden belirlenen FP/FN örneklemini oluşturun. Aşağıdaki ayar dil başına en fazla 100 FP ve 100 FN seçer; daha az hata varsa satır çoğaltılmaz ve gerçek uygunluk sayısı raporlanır:

```bash
python scripts/sample_linguistic_errors.py \
  artifacts/error_annotation/xlmr_base_surface_ensemble_predictions.csv \
  --outcomes FP,FN \
  --per-cell 100 \
  --seed 42
```

3. `annotation_blind.csv` dosyasının iki bağımsız kopyasını oluşturun. Anotatörlere yalnız kendi kopyalarını verin; `annotation_key.csv` araştırmacıda ve gizli kalmalıdır.
4. Her örneği birbirinden bağımsız en az iki uzman kodlasın.
5. Uzmanlar arası uyumu **uzlaşmadan önce** hesaplayın:

```bash
python scripts/calculate_annotation_agreement.py \
  annotation_annotator_A.csv annotation_annotator_B.csv \
  --labels annotator_A,annotator_B
```

6. `disagreements_for_adjudication.csv` içindeki uyuşmazlıkları üçüncü değerlendirici veya ortak karar ile çözün ve tekil `annotation_adjudicated.csv` oluşturun.
7. Anahtar ancak kör kodlama bittikten sonra açılmalı ve sonuçlar şu komutla özetlenmelidir:

```bash
python scripts/summarize_error_annotations.py \
  annotation_adjudicated.csv annotation_key.csv
```

## Genel kodlama kuralı

Olgular için şu değerleri kullanın:

- `1`: olgu açıkça var,
- `0`: olgu yok,
- `?`: metin/anotatör bilgisiyle karar verilemiyor.

Birden fazla olgu aynı örnekte bulunabilir. `primary_phenomenon`, model kararını etkilemesi en muhtemel tek olguyu içerir. `confidence_1_to_5`, anotatörün dilsel kodlamaya güvenidir; model tahminine güven değildir.

## Olgu tanımları

### `inflectional_suffix`

Sınıf yorumu için önemli görünen çekimsel morfoloji vardır: kişi, sayı, durum, iyelik, zaman, kip, görünüş veya olumsuzluk işaretlemesi gibi. Yalnızca dilin çekimli olması yeterli değildir; olgunun örnekte yorum açısından belirgin olması gerekir.

### `derivational_suffix`

Bir kökten yeni sözcük veya saldırgan/kimlik belirten ifade oluşturan yapım süreci vardır. Çekim ekleri bu alana yazılmaz.

### `negation`

Olumsuzluk, toksik ifadenin kapsamını değiştirir. Örneğin saldırgan bir eylemi reddetme/kınama ile doğrudan saldırıyı ayırmak için önemlidir.

### `quotation_or_reported_speech`

Saldırgan ifade konuşmacının doğrudan sözü değil, alıntı, haber, hikâye, eleştiri veya raporlanan konuşma içinde geçer.

### `target_is_explicit`

Saldırının yöneldiği kişi/grup metinde dilsel olarak açıktır. Yalnız küfür sözcüğünün bulunması hedefin açık olduğu anlamına gelmez.

### `profanity_without_abuse`

Küfür/argo vardır fakat belirli bir kişiye veya gruba saldırı işlevi taşımayabilir; ünlem, samimi konuşma, vurgu veya öz-alıntı olabilir.

### `obfuscation`

Sözcük yıldız, rakam, noktalama, boşluk, karakter tekrarı veya yazım bozma yoluyla gizlenmiştir.

### `compound_or_spacing`

Birleşik yazım, ayrı yazım veya dil-özel boşluklandırma modelin alt-sözcük analizini etkileyebilir.

### `irony_or_reclaimed_use`

İroni, mizah veya grubun bir ifadeyi geri sahiplenmesi nedeniyle yüzey sözcükleri ile gerçek pragmatik işlev ayrışır.

### `annotation_ambiguous`

Bağlam yetersizliği veya görev tanımının belirsizliği nedeniyle gold etiket makul biçimde tartışmalıdır. Bu alan “modelle aynı fikirde değilim” anlamında otomatik olarak seçilmemelidir; kısa gerekçe `notes` alanına yazılmalıdır.

### `domain_or_cultural_reference`

Kararın anlaşılması özel kişi, olay, argo, topluluk normu veya kültürel bilgi gerektirir.

## Raporlama

Her dil ve olgu için en az şu değerler raporlanmalıdır:

- her dilde mevcut ve örneklenen FP/FN sayısı,
- FP ve FN içindeki olgu prevalansı ve karar verilemeyen örnek sayısı,
- iki anotatörün ham uyumu ve Cohen's κ değeri,
- uzlaşma sonrasında baskın olgu dağılımı,
- morphology-aware yöntem ile surface baseline arasındaki **tam test kümesi üzerindeki** eşleştirilmiş fark ve %95 güven aralığı.

FP/FN-odaklı örneklemden genel-popülasyon hata oranı hesaplanmamalıdır; bunun için TP/TN dahil olasılıklı bir örnekleme tasarımı gerekir. Az sayıda örneğe sahip olgular için güçlü genelleme yapılmamalı; nitel örnekler yalnız nicel tablonun açıklaması olarak kullanılmalıdır.
