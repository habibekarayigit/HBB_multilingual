# Hakem denetimine uygun confirmatory Colab akışı

Ana dosya: [`Cross_lingual_REVISED.ipynb`](../Cross_lingual_REVISED.ipynb)

## Önemli sınır

Bu kod, dergi kabulünü garanti etmez. Ama eksik seed, değişebilir model sürümü, sahte Macarca koşulu, test-alt-kümesi seçimi ve notebook-state kaynaklı sonuç karışmasını engelleyecek şekilde hazırlanmıştır. Eski `Cross_lingual.ipynb` keşif amaçlıdır ve makale sonucu üretmek için kullanılmamalıdır.

## Deney sayısı neden 30?

Hedef diller ayrı eğitim işleri değildir. Her model Türkçe eğitim verisiyle bir kez eğitilir ve aynı checkpoint bütün test dillerinde değerlendirilir.

| Blok | Hesap | Eğitim işi |
|---|---:|---:|
| Birincil surface modeller | 4 model × 3 seed | 12 |
| Morfoloji mekanizması | 3 ek görünüm × 3 seed | 9 |
| Fince uzunluk duyarlılığı | 2 ek uzunluk × 3 seed | 6 |
| Gerçek frozen encoder | 1 koşul × 3 seed | 3 |
| **Toplam** |  | **30** |

Surface XLM-R-base `L=128` sonucu birincil blokta zaten üretildiği için diğer bloklarda tekrar edilmez.

## Baştan sona sıra

1. Colab'da A100 GPU seçilir ve Drive bağlanır.
2. Mevcut çalışma dalı Drive'a klonlanır/güncellenir.
3. `requirements-colab.txt` kurulur; Git commit, `pip freeze` ve `nvidia-smi` kaydedilir.
4. ATC ve RTP-LX denetlenir; BEEP! ve TweetEval sabit Git commitlerinden alınır.
5. Macarca dosya yalnız tam kaynak, lisans ve etiket semantiği doğrulanmışsa `HU_FILE` ile eklenir. Aksi halde koşul yok sayılmaz; açıkça dışarıda bırakılır.
6. Tokenizasyon, kırpılma, overlap ve doğrusal kontroller yeniden üretilir.
7. Stanza ile lemma/UD görünümleri üretilir. Randomized placebo, train/dev/test veya veri kümeleri arasında bilgi taşımaması için her `language/source_dataset/split` bölmesinde ayrı karıştırılır.
8. Dil başına 100 satırlık morfoloji kalite-kontrol formu hazırlanır. Kontrol tamamlanmadan `MORPH_QC_APPROVED=True` yapılmaz.
9. Hugging Face `main` etiketleri `pin_model_revisions.py` ile değişmez commit hashlerine çevrilir.
10. Preflight; 30 koşulu, seedleri, etkin batch büyüklüğünü, veri sütunlarını, dilleri, dosya hashlerini, GPU'yu ve model hashlerini doğrular.
11. GPU blokları çalıştırılır. Kesilen oturumda yalnız dört zorunlu çıktısı da bulunan koşullar atlanır; yarım koşullar yeniden çalıştırılır.
12. Üç seed tamamlanmadan birleştirme yapılmaz.
13. Sonuçlar mean±SD olarak raporlanır. Önceden tanımlı karşılaştırmalar seed ve örnek düzeyinde hiyerarşik paired bootstrap ile hesaplanır; tüm test ailesi için Holm düzeltmesi verilir.
14. Üç seed olasılıkları birleştirilerek XLM-R-base ensemble tahminleri hazırlanır. Dil başına en fazla 100 FP ve 100 FN kör örneklenir.
15. İki uzman bağımsız kodlama yapar; uzlaşmadan önce ham uyum ve Cohen's κ hesaplanır. Uyuşmazlıklar ayrı dosyada çözülür.

## Zorunlu manuel kapılar

Aşağıdakiler otomatikleştirilemez ve insan kararı gerektirir:

- Macarca kaynağın gerçekten eski çalışmada kullanılan veri olup olmadığının doğrulanması;
- Stanza analizlerinin dil uzmanlarınca kalite kontrolü;
- iki bağımsız hata anotasyonu ve uyuşmazlıkların çözülmesi;
- sonuçların nedensel/typolojik iddia gücünün iç değerlendirmesi;
- makaledeki `WORKING-DRAFT` notunun ancak tüm koşullar tamamlandıktan sonra kaldırılması.

## Üretilen temel çıktılar

- `artifacts/model_revisions.json`: model commitleri
- `artifacts/confirmatory_preflight.json`: çalıştırma öncesi doğrulama
- `outputs/encoder/<run_id>/run_config.json`: koşul ve ortam bilgisi
- `outputs/encoder/<run_id>/history.json`: gelişim kümesi geçmişi
- `outputs/encoder/<run_id>/metrics.csv`: test metrikleri
- `outputs/encoder/<run_id>/predictions.csv`: örnek düzeyi olasılıklar
- `artifacts/results/encoder_results.xlsx`: tam üç-seed özetleri
- `artifacts/results/encoder_paired_bootstrap.xlsx`: eşleştirilmiş farklar ve güven aralıkları
- `artifacts/results/figures/`: confirmatory model şekilleri
- `artifacts/error_annotation/`: kör anotasyon, uyum ve uzlaşma çıktıları

## Eski notebooktan bilinçli olarak çıkarılanlar

Legacy kodundaki Gemma, translation, tek-seed LaBSE/MPNet, rastgele 200 örneklik alt kümeler ve doğrulanamayan Macarca sonuçları confirmatory matrise taşınmamıştır. Bunları yalnız tabloyu büyütmek için geri eklemek, ana morfoloji hipotezini güçlendirmez ve karşılaştırılabilirliği bozar. Yeni deney eklenmesi gerekiyorsa önce hipotez, split, seed, metrik ve karşılaştırma kuralı önceden tanımlanmalıdır.
