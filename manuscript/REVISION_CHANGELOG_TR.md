# Makale revizyon değişiklik günlüğü

## Durum

`REVISED_MANUSCRIPT_WORKING_DRAFT.md`, yalnız depoda yeniden üretilebilen sayıları içerir. Henüz gönderime hazır değildir; GPU encoder deneyleri ve Macarca kaynak dosyası eksiktir.

## Başlıca değişiklikler

1. **Görev adı düzeltildi:** “hate speech” yerine veri kümelerinin farklı yapılarını kapsayan “abusive/offensive-language detection” kullanıldı.
2. **Başlık yumuşatıldı:** nedensel tipoloji kanıtı iddiası çıkarıldı; morphology, script ve veri karşılaştırılabilirliği birlikte ele alındı.
3. **Macarca geçici olarak çıkarıldı:** kullanılan 655 satırlık dosya depoda olmadığı için hash/split doğrulanamıyor.
4. **ATC düzeltildi:** 30.358 satır; 10.532 pozitif. Tekrarlar ve çelişkili gruplar splitten önce çıkarıldı. Çalışma splitleri 5.275/2.725, 659/341 ve 659/341 oldu.
5. **Finlandiya düzeltildi:** `>=3.5`, tamsayı sütunda `>=4` ile aynı; 392 pozitif ve 707 negatif. Eski 157 sayısı yalnız test pozitif sayısıydı.
6. **İngilizce düzeltildi:** TweetEval resmi train/dev/test toplamı 14.100; eski 11.916 yalnız train splitiydi.
7. **Korece protokol şeffaflaştırıldı:** resmi test etiketi açık olmadığı için 471 satırlık public dev dosyası test olarak kullanıldı; 1.000 satır adaptation pool olarak ayrıldı.
8. **Tokenizer denetimi eklendi:** FI testinin %99,1'i L=128'de kesiliyor ve ortalama yalnız %39 token tutuluyor.
9. **Token-overlap analizi eklendi:** TR train tarafından kapsanan hedef token occurrence oranı KO %11,1, FI %49,6, EN %64,8.
10. **Yeni yüzeysel morfoloji kontrolü eklendi:** word, char, prefix ve suffix TF-IDF modelleri aynı L=128 koşulunda karşılaştırıldı.
11. **Bootstrap eklendi:** 5.000 paired bootstrap tekrarıyla model farklarının %95 güven aralıkları hesaplandı.
12. **Eski encoder/Gemma sonuçları çıkarıldı:** clean pipeline ve doğru splitlerle yeniden üretilmediği için Tablo 9–17'deki performans iddiaları yeni taslağa aktarılmadı.
13. **Spearman tipoloji tablosu çıkarıldı:** 4 hedef/dil düzeyinde anlamlı tipolojik çıkarım sağlamıyor.
14. **Gerçek frozen baseline kodlandı:** yeni runner encoder parametrelerini gerçekten donduruyor ve trainable/frozen parametre sayılarını logluyor.

## Eski sonuç tablolarının durumu

| Eski tablo | İşlem |
|---|---|
| Table 3 Dataset overview | Tamamen düzeltildi |
| Table 4 Splits | Tamamen düzeltildi |
| Tables 5–8 Methods | Yeni audit/morphology protokolüne göre yeniden yazılmalı |
| Table 9 Monolingual | Çıkarıldı; yeniden çalıştırılacak |
| Table 10 Zero-shot | Çıkarıldı; yeniden çalıştırılacak |
| Table 11 Gemma | Çıkarıldı; bağımsız artefakt yok |
| Table 12 Baselines | Çıkarıldı; frozen sonuçları hatalı karışımdı |
| Tables 13–15 | Çıkarıldı; temiz kod/çıktı yok |
| Table 16 Correlation | Kalıcı olarak çıkarılması önerildi |
| Table 17 Prior comparison | Yeni doğrulanmış model sonuçlarından sonra hazırlanacak |

## Yeni doğrulanmış sonuç tabloları

- Düzeltilmiş veri ve split tablosu
- Tokenization/truncation tablosu
- TR→target observed-token coverage tablosu
- L=128 lexical/character/prefix/suffix diagnostic tablosu
- Paired-bootstrap farkları

## Gönderim öncesi zorunlu kalan işler

1. Macarca kaynak dosyasını eklemek ve audit etmek.
2. Stanza/UD morphology view üretmek; her dilde en az 100 örneği manuel kalite kontrolünden geçirmek.
3. `configs/experiment_matrix.json` içindeki model revision değerlerini immutable Hugging Face commit hashlerine sabitlemek.
4. 30-run encoder matrisini A100 üzerinde tamamlamak.
5. Eksik veya başarısız seed bırakmadan sonuçları otomatik birleştirmek.
6. Encoder tahminlerinden kör dilbilimsel hata anotasyonu yapmak.
7. Yeni sonuçlara göre Abstract, Results, Discussion ve Conclusion bölümlerini son kez güncellemek.
