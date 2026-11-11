# Dilbilimsel hata anotasyonu kısa kılavuzu

## Amaç

Bu anotasyon modelin tahminini yeniden etiketlemek için değil, doğru ve yanlış tahminlerle ilişkili dilsel olguları belirlemek içindir. Anotatör model adı, tahmin, güven skoru ve FP/FN bilgisini görmemelidir.

## İş akışı

1. `predictions.csv` en az `example_id, language, text, label, prediction` sütunlarını içermelidir.
2. Kör örneklem oluşturun:

```bash
python scripts/sample_linguistic_errors.py predictions.csv \
  --model xlm-roberta-base \
  --per-cell 25 \
  --seed 42
```

3. Anotatörlere yalnız `annotation_blind.csv` verin. `annotation_key.csv` araştırmacıda kalmalıdır.
4. Her örneği birbirinden bağımsız en az iki uzman kodlasın.
5. Anotasyon bittikten sonra uzmanlar arası uyum hesaplanmalı; uyuşmazlıklar üçüncü bir değerlendirici veya ortak karar ile çözülmelidir.

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

- anotasyon sayısı,
- olgu prevalansı,
- FP ve FN içindeki prevalans,
- olgu mevcut/yok iken hata oranı,
- iki anotatörün uyumu,
- morphology-aware yöntem ile surface baseline arasındaki eşleştirilmiş fark ve %95 güven aralığı.

Az sayıda örneğe sahip olgular için güçlü genelleme yapılmamalı; nitel örnekler yalnız nicel tablonun açıklaması olarak kullanılmalıdır.
