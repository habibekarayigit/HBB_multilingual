# Mevcut notebook ve makale sonuçlarının yeniden üretilebilirlik denetimi

Bu belge hakem raporu değildir; depodaki dosyalar ile makaledeki yöntem/tablolar arasındaki izlenebilirliği kontrol eder. Amaç, yeni deneylerden önce hangi sonuçların yeniden çalıştırılması gerektiğini göstermektir.

## Kısa sonuç

Mevcut `Cross_lingual.ipynb`, temiz bir runtime üzerinde baştan sona çalıştırıldığında makaledeki sonuç tablolarını üretmez. `SOTA_FINAL_V4_PLUS.xlsx` yalnızca 18 baseline satırı içerir ve makaledeki Tablo 12 ile dahi bütünüyle uyuşmaz. Bu nedenle tüm ana sonuçlar yeni, temiz ve sürüm-kilitli bir pipeline ile yeniden üretilmelidir.

## Makale tablosu–depo kanıtı eşlemesi

| Makale tablosu | İddia edilen deney | Depoda izlenebilir nihai çıktı | Durum |
|---|---|---|---|
| Tablo 9 | 3 seed monolingual, dört encoder | Yok | Yeniden çalıştırılmalı |
| Tablo 10 | 3 seed, dört encoder, full test | Yok | Yeniden çalıştırılmalı |
| Tablo 11 | Gemma 3 12B | Kod/çıktı yok | Yeniden çalıştırılmalı |
| Tablo 12 | N=200 baseline karşılaştırması | Kısmen `SOTA_FINAL_V4_PLUS.xlsx` | Değerler ve uygulamalar karışık; yeniden çalıştırılmalı |
| Tablo 13 | Few-shot 50/100/200 | Kod/çıktı yok | Yeniden çalıştırılmalı |
| Tablo 14 | Transfer matrisi | Kod/çıktı yok | Yeniden çalıştırılmalı |
| Tablo 15 | Joint training | Kod/çıktı yok | Yeniden çalıştırılmalı |
| Tablo 16 | Tipolojik Spearman analizi | Kod/çıktı yok | Tasarım değiştirilmeli |

## Kritik kod sorunları

### 1. İlk frozen baseline gerçekte frozen değildir

V4 hücresinde önce `mdl_frozen` oluşturulup `roberta` parametreleri donduruluyor. Hemen sonra:

```python
mdl_frozen = run_train(df_tr_train.sample(3000), epochs=1)
```

satırı tamamen yeni ve tüm katmanları eğitilebilir bir model oluşturup önceki frozen modeli siliyor. İlk hücrede raporlanan KO `0.467948` ve EN `0.337318` sonuçları geçerli bir frozen-encoder deneyi değildir.

Sonraki hücrede `run_train_frozen` doğru biçimde uygulanmış ve dört dil için `0.333333` üretilmiştir. Buna rağmen makaledeki Tablo 12, KO için ilk/geçersiz `0.468`, FI ve HU için sonraki `0.333`, EN için yine ilk `0.337` değerini birlikte kullanmaktadır. Frozen baseline bütünüyle yeniden çalıştırılmalıdır.

### 2. LaBSE dört dil sonucu nihai dosyada yok

Düzeltme hücresinde `b1_res` tanımı `return` ifadesinin arkasında ve yanlış girintidedir; dolayısıyla çalışmaz. `df_labse`, önceki hücreden bellekte kalan ve yalnız KO/EN içeren eski `b1_res` değişkenini kullanır. Bu nedenle `SOTA_FINAL_V4_PLUS.xlsx` içinde LaBSE için FI ve HU satırları yoktur; makaledeki bu iki değer mevcut artefakttan üretilemez.

### 3. Notebook temiz runtime'a bağımsız değildir

V4 içinde `df_tr_test` tanımlanmadan kullanılır. Değişken yalnız önceki büyük hücre çalıştırılmışsa bellekte bulunur. Benzer biçimde bazı sonuç listeleri önceki hücre durumundan alınır. Bu, hücrelerin farklı sırada çalıştırıldığı stateful notebook problemidir.

### 4. Ana deney yalnız tek seed ve tek encoder içerir

V4 nihai blok `MODEL_NAME = "xlm-roberta-base"` ve `set_seed(42)` kullanır. Üç seed döngüsü, mBERT, XLM-R-large veya Glot500 eğitim döngüsü yoktur. Makaledeki dört-model/üç-seed sonuçları bu notebook ile izlenebilir değildir.

### 5. Sonuç dosyası full-test ana sonuçları içermez

`SOTA_FINAL_V4_PLUS.xlsx` içindeki ZeroShot sonuçları 200 örneklik alt kümelere aittir. Bunlar makaledeki full-test Tablo 10 sonuçları değildir. Full test ve dengeli alt küme metrikleri ayrı dosya ve ayrı tablo olarak tutulmalıdır.

### 6. Tahmin fonksiyonu batch kullanmıyor

`predict_fast`, tüm metinleri tek tokenizer/model çağrısında GPU'ya yollar. Full test veya uzun Finlandiya metinlerinde OOM riski vardır. Sabit batch boyu ile döngü kullanılmalıdır.

### 7. Finlandiya metinleri ciddi biçimde kesiliyor

Yerel XLM-R tokenizer ile yaptığımız ölçümde Finlandiya testindeki 440 metnin 436'sı (%99,1) `max_length=128` altında kesilmektedir ve ortalama token-retention yalnız %39,0'dır. Karşılaştırma için kesilme oranı Türkçede %1,4, İngilizcede %0,1 ve Korecede %0'dır. Bu koşul diller arası karşılaştırmayı ciddi biçimde confound eder. 128/256/512 duyarlılık analizi ve uzunluk eşleştirilmiş Finlandiya testi zorunludur. Ayrıntılar `artifacts/tokenization/REPORT.md` dosyasındadır.

### 8. Back-translation hesaplanıyor fakat eğitime eklenmiyor

`df_back_tr` oluşturuluyor ancak `run_train` girdisine dahil edilmiyor. Bu nedenle notebook'taki back-translation kodu bir deney sonucu üretmez ve yöntem bölümünde uygulanmış deney olarak anlatılmamalıdır.

### 9. İlk büyük pipeline hata ile bitiyor

Notebook çıktısında ilk pipeline `NameError: name 'tok_tr' is not defined` ile sonlanmıştır. Kurtarma hücreleri notebook state'ine ve Drive'daki yedek dosyalara dayanır. Bu koşul yeniden üretilebilir bir final pipeline değildir.

## Veri bölme sorunları

- ATC deduplication yapılmadan rastgele bölünüyor; kopya sızıntısı mümkündür.
- İlk pipeline ATC'nin yaklaşık 24.286 satırını train olarak kullanırken V4 8.000 satır örnekler. Makaledeki hangi tablonun hangi eğitim setiyle üretildiği açık değildir.
- Kod Macarca verinin %50'sini test olarak seçer; makale 589 test örneği ve ayrı adaptation pool tanımlar. Aynı split değildir.
- Kod Korece yalnız `labeled/dev.tsv` dosyasını yükler; makale 6.396/500/1.000 train/dev/test yapısı raporlar. Split kaynakları izlenebilir değildir.
- Finlandiya full veri pozitif sayısı 392 iken makale 157 raporlar.

## Yeniden çalışma için asgari kabul ölçütleri

1. Boş Colab/runtime üzerinde tek komutla baştan sona çalışma.
2. Veri dosyası SHA-256 değerleri ve sabit split manifestleri.
3. Her tablo satırı için `run_id`, model, seed, veri split'i ve checkpoint.
4. Her sonuç için örnek-düzeyi gold/prediction/probability dosyası.
5. Frozen baseline'da eğitilebilir parametre adlarının ve sayısının loglanması.
6. Full test ile alt küme sonuçlarının kesin ayrımı.
7. Her üç seed tamamlanmadan mean±std tablosu üretmeme.
8. Notebook yerine yapılandırma dosyalı script; notebook yalnız analiz/görselleştirme amacıyla kullanılmalı.
