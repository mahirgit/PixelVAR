# PixelVAR Projesi Durum ve Sonuç Raporu

Tarih: 2026-06-07

Bu rapor, PixelVAR projesinde şu ana kadar yapılan işleri, elde edilen
sonuçları, hangi proposal maddelerinin tamamlandığını, hangi maddelerin
tamamlanamadığını ve bunların nedenlerini Türkçe ve ayrıntılı şekilde özetler.

## 1. Kısa Özet

Projede ana hedef, pixel art karakter sprite'larını doğrudan ayrık palet-token
uzayında üreten bir coarse-to-fine autoregressive model kurmaktı. Proposal'da
bu fikir PixelVAR olarak tanımlanmıştı: önce çok ölçekli bir tokenizer ile
sprite'ı 1x1'den 32x32'ye kadar token haritalarına ayırmak, sonra bir
Transformer ile bu ölçekleri kaba detaydan ince detaya doğru üretmek.

Şu ana kadar pratik olarak çalışan en güçlü model:

- Model: `var_sprites_v0_full`
- Veri: MSD Sprites replacement dataset, gerçek sprite'lar
- Çözünürlük: 32x32
- Token uzayı: şeffaf token `0` + 16 palet rengi tokenı, yani `0..16`
- Ölçekler: `[1, 2, 4, 8, 16, 32]`
- Toplam token sayısı: `1 + 4 + 16 + 64 + 256 + 1024 = 1365`
- En iyi sampling ayarı: `temperature=0.8`, `top_k=8`
- Ana sprite-feature skoru: `0.00147`
- Palette consistency: `1.0000`
- Opaque ratio: `0.2291`, referans validation `0.2354`
- Edge density: `0.1808`, referans validation `0.1811`

Bu model şu an ana sonuç olarak tutuluyor. HMAR masked-refinement branch'i
teknik olarak başarılı oldu ve Inception tabanlı bazı metriklerde ana modele
çok yakın, hatta biraz daha iyi göründü; fakat proposal içi sprite-feature
evaluator ana PixelVAR modelini seçti. Flat raster AR baseline ise sayısal
metriklerde çok iyi görünmesine rağmen memorization audit'te açık şekilde
memorize ettiği için temiz generative winner olarak kabul edilmedi.

## 2. Proposal'da Söylenenler ve Mevcut Durum

Proposal'daki ana iddialar ve bizim şu anki durumumuz:

| Proposal maddesi | Durum | Açıklama |
| --- | --- | --- |
| Dataset curation | DONE | Sprites replacement, Pokemon ve OpenGameArt yolları hazırlandı. |
| 32x32 sprite preprocessing | DONE | Bütün ana deneyler 32x32 üzerinden yürütüldü. |
| 16 renkli palet çıkarımı | DONE | K-means tabanlı 16 renkli global palet pipeline'ı çalışıyor. |
| Ayrık palet-index uzayında üretim | DONE | Ana model continuous RGB üretmiyor; doğrudan token üretiyor. |
| Multi-scale hierarchy, 1x1 to 32x32 | DONE | 6 ölçekli 1365-token pyramid kullanıldı. |
| VAR-style next-scale generation, Option A | DONE | Ana PixelVAR modeli eğitildi, değerlendirildi ve örnekler üretildi. |
| HMAR / masked refinement, Option B | DONE | Ayrı branch olarak implement edildi, eğitildi ve 1/2/4/8 refinement ablation yapıldı. |
| Raster-scan AR baseline | DONE | Implement edildi ve değerlendirildi; memorization nedeniyle temiz winner değil. |
| Flat MaskGIT baseline | DONE | Implement edildi ve değerlendirildi; mevcut ayarda başarısız oldu. |
| FID / standard image metrics | PARTIAL DONE | Inception FID, KID, precision/recall, density/coverage, MS-SSIM evaluator eklendi ve iç modellerde çalıştırıldı. Dış baselinelar için henüz tam kullanılmadı. |
| Palette Consistency Score | DONE | Evaluator içinde var ve ana modellerde `1.0000`. |
| Edge crispness / edge density | DONE | Pixel-art-specific proxy olarak edge density raporlanıyor. |
| Multi-scale VQ-VAE tokenizer | PARTIAL / NOT DONE | Neural VQ-VAE implement edildi ama kalite yetersiz olduğu için ana yol olmadı. Patch-VQ alternatifi denendi. Proposal'daki tam çok ölçekli VQ-VAE hedefi tamamlanmadı. |
| Codebook size 8/16/32 ablation | NOT DONE | Sistematik 8/16/32 ablation yapılmadı. Ana sonuç 16 renk. |
| Number-of-scales ablation | NOT DONE | Sistematik ölçek sayısı ablation'ı yapılmadı. |
| PixDiff-PIG baseline | NOT DONE | Çalıştırılmadı. |
| SD 1.5 LoRA + quantization baseline | NOT DONE | Henüz implement edilmedi. |
| SD-piXL external baseline | PARTIAL DONE | Repo-side setup, Modal action ve normalization pipeline hazır; actual smoke/batch run henüz yapılmadı. |
| User study, n >= 20 | NOT DONE | Yapılmadı. |
| Final report / presentation | PARTIAL DONE | Final rapor artifact'ları var; presentation deck henüz hazırlanmadı. |

## 3. Veri Tarafında Ne Yapıldı?

### 3.1 Orijinal Sprites Dataset Sorunu

Proposal'da Sprites dataset yaklaşık `170K` frame olarak planlanmıştı. İlk
hedef orijinal Kaggle kaynağını kullanmaktı. Ancak pratikte şu problem çıktı:

- `brentspell/sprites-dataset` Kaggle API ile dosya listeleme sırasında `403
  Forbidden` verdi.
- Tarayıcıda dataset sayfası da bulunamadı.
- Bu yüzden aynı yönde kullanılabilecek en yakın canlı replacement arandı.

Ana replacement olarak `TalBarami/msd_sprites` kullanıldı. Bu, YingzhenLi
Sprites ailesine yakın, canlı ve erişilebilir bir varyant olarak seçildi.

### 3.2 Ana Sprites Replacement Verisi

İşlenen ana dataset:

- Source: `TalBarami/msd_sprites`
- Curated frame sayısı: `93,312`
- Train: `74,664`
- Validation: `9,360`
- Test: `9,288`
- Grouping rule: `body`, `bottom`, `top`, `hair`
- Transparency handling: köşelerden bağlı siyah background alpha'ya çevrildi.

Bu split özellikle önemliydi. Pixel art sprite dataset'lerinde aynı karakterin
çok benzer varyantları train ve validation'a sızarsa metrikler yapay olarak
iyileşebilir. Bu yüzden mümkün olduğunca group-safe split uygulandı.

### 3.3 Pokemon Verisi

Pokemon sprites daha erken baseline/deneme olarak hazırlandı ve eğitildi.
Sonuç:

- Run: `var_pokemon_v0_full`
- Best validation loss: `0.34376`
- Best validation accuracy: `0.90941`
- Epoch: `31`

Pokemon tarafı ana sonuç olmadı; ana sonuç sprites replacement üzerinde kaldı.

### 3.4 OpenGameArt Verisi

OpenGameArt, proposal dışı ama mantıklı bir multi-dataset stretch olarak eklendi.
Burada amaç, gerçek sprite çeşitliliğini artırmanın modeli iyileştirip
iyileştirmediğini test etmekti.

OpenGameArt curated dataset:

- Frame sayısı: `4,659`
- Group sayısı: `105`
- Train: `3,493`
- Validation: `502`
- Test: `664`
- Public OpenGameArt asset sayısı: `7`
- Kaynakların çoğu: `CC0`

Daha sonra bu veri, real sprites ve generated-keep verisi ile karıştırıldı.

## 4. Tokenizer ve Preprocessing

### 4.1 Ana Tokenizer: Deterministic Palette Pyramid

Proposal'da multi-scale VQ-VAE tokenizer hedeflenmişti. Ancak pratikte önce
deterministic palette-token pipeline kuruldu. Bu kararın nedeni:

1. Modelin geri kalanı için net ve güvenilir bir discrete-token contract
   gerekiyordu.
2. VQ-VAE kalite riski yüksekti; tokenizer kötü olursa model ne kadar iyi
   olursa olsun output kötü olacaktı.
3. Projenin zaman sınırı içinde önce çalışan ana VAR hattını çıkarmak daha
   mantıklıydı.

Deterministic tokenizer:

- 32x32 RGBA sprite alır.
- Opaque pixelleri 16 renkli global palete quantize eder.
- Şeffaf pixeller token `0` olur.
- Palet renkleri token `1..16` olur.
- Multi-scale pyramid, mode pooling ile oluşturulur.

Ölçekler:

| Ölçek | Token sayısı |
| ---: | ---: |
| 1x1 | 1 |
| 2x2 | 4 |
| 4x4 | 16 |
| 8x8 | 64 |
| 16x16 | 256 |
| 32x32 | 1024 |
| Toplam | 1365 |

Bu yol proposal'daki "palette constrained discrete generation" iddiasını güçlü
şekilde karşıladı, ancak proposal'daki "train a VQ-VAE tokenizer" maddesini
tam olarak karşılamadı.

### 4.2 Neural VQ-VAE Denemesi

Neural VQ-VAE implement edildi:

- Encoder/decoder path yazıldı.
- VQ codebook path yazıldı.
- Eğitim script'i eklendi.
- 8x8 ve 16x16 latent varyantları denendi.

Ancak görüntü kalitesi yeterli olmadı:

- Reconstruction'lar fazla soft/ghosted göründü.
- Pixel art için crisp kenarlar kayboldu.
- Code usage zayıftı; codebook yeterince verimli kullanılmadı.
- Bu tokenizer ile devam etmek ana modeli riske atacaktı.

Bu yüzden neural VQ-VAE ana Stage 1 olarak promote edilmedi. Bu, proposal'daki
en büyük teknik sapmadır.

### 4.3 Patch-VQ Alternatifi

Learned-token fikrini tamamen bırakmamak için patch-VQ alternatifi eklendi.
Patch-VQ, 2x2 RGBA patch'leri üzerinden KMeans benzeri bir codebook öğreniyor.

Patch-VQ sonuçları:

- Dataset: `data/processed/sprites_patchvq16`
- Source: real MSD Sprites
- Token map shape: `(93,312, 16, 16)`
- Vocab size: `512`
- Used codes: `143/512`
- Train: `74,664`
- Val: `9,360`
- Test: `9,288`

Patch-VQ teknik olarak çalıştı:

- Token export çalıştı.
- VAR, learned token map üzerinde eğitildi.
- Decoded sample'lar coherent ve recognizable oldu.

Ama kalite ana palette-token VAR'dan düşük kaldı:

- Görseller daha blocky.
- Stil daha bias'lı.
- Detay kalitesi ana modelden zayıf.

Bu yüzden Patch-VQ "learned-token ablation" olarak kaldı, ana sonuç olmadı.

## 5. Ana Model: PixelVAR Option A

Ana model proposal'daki VAR-style next-scale prediction fikrini pratik olarak
gerçekleştiriyor.

Model davranışı:

- Coarser scale tokenları context olarak alıyor.
- Target scale tokenlarını paralel tahmin ediyor.
- Üretim coarse-to-fine ilerliyor.
- Her token için softmax vocabulary `0..16`.

Ana training ladder:

| Run | Amaç | Sonuç |
| --- | --- | --- |
| `var_sprites_overfit32` | Memorization gate | final `train_loss=0.04542`, `train_acc=0.98361` |
| `var_sprites_debug1k` | Sanity run | best `val_loss=0.12671`, `val_acc=0.96116`, epoch 90 |
| `var_sprites_v0_full` | Full replacement dataset | best `val_loss=0.01991`, `val_acc=0.99223`, epoch 14 |

### 5.1 Sampling Sweep

Ana checkpoint için sampling sweep:

- Temperature: `0.6`, `0.8`, `1.0`
- Top-k: `8`, `16`, `none`
- Her ayar: `128` sample
- Referans: `2,048` validation sprite

En iyi ayar:

- `temperature=0.8`
- `top_k=8`
- Feature FID-style score: `0.00147`
- Palette consistency: `1.0000`
- Opaque ratio: `0.2291`
- Reference opaque ratio: `0.2354`
- Edge density: `0.1808`
- Reference edge density: `0.1811`

Interpretation:

- Model, validation set'in ortalama doluluk oranına ve edge/detail yoğunluğuna
  çok yakın sprite'lar üretiyor.
- Çıktılar palette-consistent; yani post-hoc quantization gerekmiyor.
- Görsel sample sheet'ler genel olarak coherent, crisp ve sprite formatına uygun.

## 6. 170K Generation Hedefi

Proposal'daki veri ölçeği ve üretim hedefi açısından büyük generation pass
yapıldı.

İlk büyük set:

- Sample sayısı: `8,192`
- Temperature: `0.8`
- Top-k: `8`
- Seed: `42`
- Token arrays: `(8192, 1365)`, uint8
- Final maps: `(8192, 32, 32)`, uint8

Inspection:

- Total: `8,192`
- Keep: `7,782`
- Review: `410`
- Reject: `0`

Sonra proposal-scale generation:

- Sample sayısı: `170,000`
- Temperature: `0.8`
- Top-k: `8`
- Seed: `42`
- Opaque ratio mean: `0.2276`
- Edge density mean: `0.1803`
- Token range: `0..16`

Inspection/filter sonucu:

- Keep: `161,479`
- Review: `8,500`
- Reject: `21`

Bu sonuç önemli çünkü model sadece küçük demo üretmedi; büyük ölçekli sample
generation da çalıştı. Reject oranı çok düşük kaldı. Review bucket çoğunlukla
wide-arm veya sıra dışı silhouette örneklerinden oluştu; tamamen bozuk output
oranı düşüktü.

## 7. Generated-Keep ve Mixed Training Deneyleri

### 7.1 Generated-Keep Training

170K generated sample içinden filtrelenen clean set normal processed dataset
olarak import edildi.

Generated-keep dataset:

- Sample: `161,479`
- Train: `129,183`
- Val: `16,148`
- Test: `16,148`

Training:

| Run | Amaç | Sonuç |
| --- | --- | --- |
| `var_sprites_generated_keep_overfit32` | Memorization gate | final `train_loss=0.01814`, `train_acc=0.99528` |
| `var_sprites_generated_keep_debug1k` | Sanity run | best `val_loss=0.15800`, `val_acc=0.95091`, epoch 95 |
| `var_sprites_generated_keep_v0_full` | Full generated-keep pass | best `val_loss=0.06441`, `val_acc=0.97924`, epoch 18 |

Evaluation:

- Best setting: `temperature=0.8`, `top_k=8`
- Feature score: `0.00157`
- Palette consistency: `1.0000`
- Opaque ratio: `0.2181`
- Generated-keep validation opaque ratio: `0.2241`
- Edge density: `0.1761`
- Generated-keep validation edge density: `0.1772`

Bu sonuç, generated data üzerinde modelin öğrenebildiğini gösterdi. Ancak bu
skor kendi generated-validation referansına göre olduğu için doğrudan real-only
winner comparison olarak kullanılmadı.

### 7.2 Real + Generated Mixed Training

Real sprites ile generated-keep set karıştırıldı.

Mixed dataset:

- Real Sprites: `93,312`
- Generated-keep: `161,479`
- Total: `254,791`
- Train: `203,847`
- Val: `25,508`
- Test: `25,436`

Training:

| Run | Amaç | Sonuç |
| --- | --- | --- |
| `var_sprites_mixed_overfit32` | Memorization gate | final `train_loss=0.00681`, `train_acc=0.99654` |
| `var_sprites_mixed_debug1k` | Sanity run | best `val_loss=0.12671`, `val_acc=0.96116`, epoch 90 |
| `var_sprites_mixed_v0_full` | Full mixed pass | best `val_loss=0.05028`, `val_acc=0.98388`, epoch 14 |

Bu run bir kez local 2-hour command limit yüzünden epoch 3 civarında kesildi.
Modal checkpoint volume'da `last.ckpt` olduğu için resume edildi. Daha sonra
Modal spend limit artırıldıktan sonra tamamlandı.

Real-validation evaluation:

- Best setting: `temperature=0.8`, `top_k=8`
- Feature score: `0.00755`
- Palette consistency: `1.0000`
- Opaque ratio: `0.2220`, real validation `0.2354`
- Edge density: `0.1797`, real validation `0.1811`

Sonuç: Mixed model teknik olarak çalıştı ama real-only ana modelden kötü kaldı.
Bu yüzden ana sonuç yapılmadı.

### 7.3 Real + Generated + OpenGameArt Mixed Training

OpenGameArt da mixed dataset'e eklendi.

Dataset:

- Real Sprites: `93,312`
- Generated-keep: `161,479`
- OpenGameArt: `4,659`
- Total: `259,450`
- Train: `207,340`
- Val: `26,010`
- Test: `26,100`

Training:

| Run | Amaç | Sonuç |
| --- | --- | --- |
| `var_sprites_mixed_oga_overfit32` | Memorization gate | final `train_loss=0.007`, `train_acc=0.997` |
| `var_sprites_mixed_oga_debug1k` | Sanity run | final `val_loss=0.127`, `val_acc=0.961` |
| `var_sprites_mixed_oga_v0_full` | Full OpenGameArt-mixed pass | best `val_loss=0.05408`, `val_acc=0.98295`, epoch 14 |

Real-validation evaluation:

- Best setting: `temperature=0.8`, `top_k=8`
- Feature score: `0.00778`
- Palette consistency: `1.0000`
- Opaque ratio: `0.2200`, real validation `0.2354`
- Edge density: `0.1777`, real validation `0.1811`

Sonuç: OpenGameArt path reproducible ve çalışır hale geldi, fakat sample-quality
metriklerinde ana real-only modelden kötü kaldı.

## 8. HMAR / Masked Refinement Sonuçları

Proposal'da Option B olarak HMAR-style masked refinement denenmesi planlanmıştı.
Bu tamamlandı.

HMAR model:

- Aynı coarse-to-fine ölçek sırasını koruyor.
- Her scale içinde target tokenlar maskeleniyor.
- Reserved mask token: `17`
- Model tüm target scale'i paralel tahmin ediyor.
- Sampling sırasında confidence'a göre bazı tokenlar tutuluyor, kalanlar tekrar
  maskeleniyor.

HMAR training:

| Run | Amaç | Sonuç |
| --- | --- | --- |
| `hmar_sprites_overfit32` | Memorization gate | final `train_loss=0.02308`, `train_acc=0.98925` |
| `hmar_sprites_debug1k` | Sanity run | final `val_loss=0.14522`, `val_acc=0.95268` |
| `hmar_sprites_v0_full` | Full HMAR run | early stopped at epoch 13, `val_loss=0.02025`, `val_acc=0.99216` |

Refinement-step ablation:

| Model | Steps | Temperature | Top-k | Feature score | Opaque ratio | Edge density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VAR baseline | - | `0.8` | `8` | `0.00147` | `0.2291` | `0.1808` |
| HMAR | `1` | `0.8` | `8` | `0.00189` | `0.2315` | `0.1833` |
| HMAR | `4` | `0.8` | `16` | `0.00366` | `0.2264` | `0.1823` |
| HMAR | `2` | `0.8` | `16` | `0.00374` | `0.2297` | `0.1831` |
| HMAR | `8` | `1.0` | `none` | `0.00822` | `0.2191` | `0.1747` |

Interpretation:

- HMAR implementasyonu başarılı.
- En iyi HMAR ayarı sadece `1` refinement step.
- Daha fazla refinement bu düşük çözünürlükte kaliteyi artırmadı.
- 8 step özellikle underfilled sprite'lara ve validation statistics'ten
  uzaklaşmaya yol açtı.
- Bu yüzden HMAR ana modelin yerine geçirilmedi; proposal ablation olarak
  raporlanmalı.

## 9. Baseline Sonuçları

### 9.1 Flat Raster AR

Flat raster-scan autoregressive Transformer implement edildi. Bu model
hierarchy kullanmadan 32x32 final token grid'i raster order ile üretir.

Known image-generation metrics'te çok iyi göründü:

| Method | FID | KID | Precision | Recall | Density | Coverage | Exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat raster AR | `9.3558` | `0.002975` | `0.9172` | `0.9541` | `0.7654` | `0.8518` | `0.0112` |

Ancak memorization audit sonucu kötü:

- Train exact matches: `3162 / 4096`
- Val exact matches: `365 / 4096`
- Test exact matches: `333 / 4096`
- Generated duplicates: `206`

Processed dataset içinde cross-split exact duplicate yoktu. Bu yüzden held-out
exact match'ler basit split duplicate leakage ile açıklanamıyor. Flat AR,
sayısal metriklerde iyi görünse de temiz generative winner olarak kabul
edilmedi.

### 9.2 Flat MaskGIT

Flat MaskGIT baseline implement edildi ve eğitildi. Sonuç mevcut setup'ta
başarısız:

- FID: `67.3908`
- KID: `0.061825`
- Precision: `0.1545`
- Recall: `0.0317`
- Coverage: `0.0454`
- MS-SSIM: `0.9426`

Yorum:

- Low diversity ve poor coverage var.
- Bu ayarda Flat MaskGIT rekabetçi değil.

### 9.3 Inception-Based Known Metrics

4096 sample ile image-folder evaluator çalıştırıldı.

| Method | FID lower | KID lower | Precision higher | Recall higher | Density higher | Coverage higher | MS-SSIM lower | Exact match lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat raster AR | `9.3558` | `0.002975` | `0.9172` | `0.9541` | `0.7654` | `0.8518` | `0.8281` | `0.0112` |
| HMAR step=1 | `12.6555` | `0.005209` | `0.6494` | `0.9314` | `0.4136` | `0.6165` | `0.8309` | `0.0005` |
| PixelVAR main | `13.2516` | `0.006010` | `0.6406` | `0.9292` | `0.4093` | `0.6050` | `0.8323` | `0.0007` |
| Flat MaskGIT | `67.3908` | `0.061825` | `0.1545` | `0.0317` | `0.0455` | `0.0454` | `0.9426` | `0.0000` |

Bu tabloyu raporda dikkatli kullanmak gerekir. Sadece sayılara bakarsak Flat AR
en iyi gibi görünür. Ancak memorization audit ile bu sonuç diskalifiye edilmeli
veya en azından güçlü uyarı ile verilmelidir. Audit-adjusted bakışta en güçlü
iki aday PixelVAR main ve HMAR step=1'dir.

## 10. Memorization Audit

Memorization audit, generated sample'ları train/val/test split'lerindeki 32x32
palette-token map'lerle exact-match seviyesinde karşılaştırdı.

4096 generated sample üzerinden:

| Model | Train exact | Val exact | Test exact | Generated duplicates | Durum |
| --- | ---: | ---: | ---: | ---: | --- |
| PixelVAR main | `157` | `13` | `11` | `4` | Düşük exact-match; disclose and keep |
| HMAR step=1 | `154` | `15` | `22` | `5` | Düşük exact-match; disclose and keep |
| Flat AR | `3162` | `365` | `333` | `206` | Memorizing; clean winner değil |

Yorum:

- PixelVAR ve HMAR'da exact match sıfır değil. Bu dürüstçe raporlanmalı.
- Ancak Flat AR ile karşılaştırıldığında davranış tamamen farklı.
- PixelVAR/HMAR düşük exact-match oranında kalırken Flat AR train set'in büyük
  kısmını birebir yeniden üretme eğiliminde.

## 11. External Baseline Durumu

Proposal'da external continuous baselines vardı:

- PixDiff-PIG
- SD 1.5 LoRA + quantization
- Continuous diffusion / LoRA tarzı pratik generator

Bunlar henüz çalıştırılmadı.

Yaptığımız şey:

- External image-folder protocol yazıldı.
- `scripts/evaluate_image_folders.py` ile FID/KID/precision/recall/density/
  coverage/MS-SSIM ve pixel-art-specific metrikler aynı klasör formatında
  hesaplanabilir hale geldi.
- SD-piXL için setup path hazırlandı:
  - `scripts/export_palette_hex.py`
  - `scripts/prepare_sd_pixl_baseline.py`
  - `scripts/normalize_external_images.py`
  - `configs/external/sd_pixl_prompts.txt`
  - Modal actions: `prepare-sd-pixl-baseline`, `run-sd-pixl-smoke`,
    `run-sd-pixl-batch`

Ancak SD-piXL actual generation henüz run edilmedi. Bunun nedeni:

- SD-piXL optimization-based bir yöntem; tek image için bile uzun sürebilir.
- Büyük Hugging Face model ağırlıkları indirir.
- Normal run birkaç saat sürebilir ve GPU credit harcar.
- 4096 sample gibi numeric FID/KID için gerekli ölçek pratikte çok pahalıdır.

Bu yüzden SD-piXL şu an "wired but not benchmarked" durumunda. İlk mantıklı
adım sadece smoke run ve qualitative sample sheet'tir.

## 12. Larger Dimensions, 64x64 ve Neden Ertelendi?

Bize daha büyük çözünürlük denemesi, örneğin 64x64, soruldu. Bunu şu anda
bilinçli olarak erteledik.

Ana nedenler:

1. Proposal'ın ana hedefi zaten 32x32 sprite üretmekti.
2. Önce proposal'da söylediğimiz ana şeyleri bitirmek gerekiyordu: dataset,
   tokenizer, VAR, HMAR, baselines, metrics.
3. 64x64'e geçmek sadece output boyutunu büyütmek değil; token sequence
   uzunluğunu ve compute cost'u ciddi artırıyor.
4. Altyapı tarafında Lightning AI / uzun run yönetimi / Modal limitleri ve
   spend limit artışları gibi pratik problemler yaşandı.
5. Daha büyük çözünürlükte training, evaluation ve generation süreleri daha
   pahalı olacaktı.

32x32 sequence:

- Ölçekler: `[1, 2, 4, 8, 16, 32]`
- Token count: `1365`
- Final scale token count: `1024`

64x64 olursa doğal ölçekler:

- Ölçekler: `[1, 2, 4, 8, 16, 32, 64]`
- Token count: `1 + 4 + 16 + 64 + 256 + 1024 + 4096 = 5461`
- Final scale token count: `4096`

Yani toplam sequence yaklaşık 4 katına çıkar. Final scale tek başına 4 katına
çıkar. Bu sadece training'i değil, sampling, evaluation, memorization audit,
image export ve generated-set storage maliyetlerini de artırır. Transformer
tarafında context length ve batch memory pressure da yükselir.

Ayrıca proposal'daki eksikler varken 64x64'e geçmek riskliydi. Önce 32x32
çözünürlükte:

- Ana model çalışıyor mu?
- HMAR işe yarıyor mu?
- Baseline'lar ne gösteriyor?
- Memorization var mı?
- External metric protocol kurulabiliyor mu?

bu soruları cevaplamak daha doğruydu. 64x64 bu yüzden "future work" olarak
ertelendi.

## 13. Lightning AI, Modal ve Altyapı Notları

Başta çalışma ortamı olarak Lightning AI düşünülmüştü. Ancak pratikte uzun GPU
job'larını, data volume'ları, checkpoint'leri ve tekrar çalıştırılabilir
komutları yönetmek için Modal B200 hattı kuruldu.

Modal tarafında:

- `pixelvar-data`
- `pixelvar-outputs`
- `pixelvar-checkpoints`

volume'ları kullanıldı.

B200 ile:

- data preparation
- training ladder
- sample generation
- evaluation
- memorization audit
- report artifact üretimi

run edilebilir hale geldi.

Yaşanan altyapı problemleri:

- Bazı uzun run'lar local command timeout nedeniyle kesildi.
- Modal spend/usage limitleri birkaç kez artırılmak zorunda kaldı.
- Mixed ve OpenGameArt-mixed run'lar resume edilerek tamamlandı.
- Bazı dataset kaynakları erişilemediği için replacement seçildi.

Bu problemler model fikrinden kaynaklanmadı; daha çok zaman, GPU credit,
çalışma ortamı ve veri erişimi problemleriydi. Yine de proje kararlarını
etkiledi: 64x64, büyük external baseline batch'leri ve user study ertelendi.

## 14. Neler Tamamlanamadı ve Neden?

### 14.1 Exact Multi-Scale VQ-VAE Tokenizer

Tamamlanmadı.

Neden:

- Neural VQ-VAE denendi ama reconstruction kalitesi pixel art için yetersizdi.
- Çıktılar soft/ghosted idi.
- Pixel art'ta crisp silhouette ve exact palette behavior kritik.
- Deterministic tokenizer daha güvenilir sonuç verdi.

Bu nedenle proposal'daki "VQ-VAE tokenizer" maddesi tam olarak sağlanmadı.
Ancak learned-token yönü Patch-VQ ile ablation olarak denendi.

### 14.2 32-Color Codebook ve 8/16/32 Ablation

Tamamlanmadı.

Neden:

- Ana model 16 renkli palette ile stabil ve iyi çalıştı.
- Zaman, GPU bütçesi ve diğer proposal maddelerini tamamlama önceliği nedeniyle
  sistematik 8/16/32 ablation yapılmadı.
- 32 renk output diversity'i artırabilir ama evaluation ve tokenizer behavior
  tekrar çalıştırılmalı.

### 14.3 Number-of-Scales Ablation

Tamamlanmadı.

Neden:

- Ana 6-scale design proposal ile uyumlu ve çalışır hale geldi.
- Farklı scale sayıları yeni config, training ve evaluation maliyeti getirir.
- Önce ana ve HMAR branch'leri bitirmek daha kritik görüldü.

### 14.4 External Baselines

Tamamlanmadı veya partial.

Durum:

- Flat AR ve Flat MaskGIT internal architecture baselines olarak tamamlandı.
- SD-piXL setup hazır ama actual run yok.
- PixDiff-PIG yok.
- SD 1.5 LoRA + quantization yok.
- MDIGAN incelenebilir ama conditional pose/imputation task olduğu için doğrudan
  unconditional sprite generation'a temiz uyarlamak zor.

Neden:

- External baseline'lar çok farklı task/protocol kullanıyor.
- SD-piXL per-image optimization olduğu için büyük sample benchmark pahalı.
- LoRA/diffusion baseline prompt ve post-processing hassas.
- Önce bizim model ve internal ablation'ların doğru çalıştığını kanıtlamak daha
  öncelikliydi.

### 14.5 User Study

Tamamlanmadı.

Neden:

- n >= 20 user study için UI/form, sample randomization, consent/collection ve
  analiz süreci gerekiyor.
- Zamanın büyük kısmı model, dataset, training, baseline ve audit'e gitti.
- User study model implementation'dan farklı bir operasyonel iş yükü.

### 14.6 64x64 veya Daha Büyük Output

Tamamlanmadı.

Neden:

- Proposal main target 32x32 idi.
- 64x64 sequence length yaklaşık 4x artıyor.
- Bütçe ve zaman maliyeti yüksek.
- Önce proposal core maddelerini bitirmek gerekiyordu.
- Lightning/Modal altyapı limitleri ve uzun run kesintileri zaten 32x32'de bile
  yönetilmesi gereken problemlerdi.

## 15. Ana Kararlar ve Teknik Gerekçeler

### 15.1 Neden Main Result Real-Only VAR?

Çünkü aynı real-validation sprite-feature evaluator üzerinde en iyi skor ana
real-only VAR modelinden geldi:

- Real-only VAR: `0.00147`
- HMAR step=1: `0.00189`
- Real + generated mixed: `0.00755`
- OpenGameArt mixed: `0.00778`
- Patch-VQ decoded: `0.04689`, ayrı evaluator

Generated-keep kendi validation'ına göre iyi görünse de doğrudan real-validation
winner comparison değildir.

### 15.2 Neden HMAR Ana Model Olmadı?

HMAR başarılı bir implementation ve çok yakın bir ablation. Ancak:

- Sprite-feature evaluator ana VAR'ı seçti.
- Fazla refinement kötüleşti.
- 1-step HMAR en iyi HMAR oldu, yani iterative refinement'in beklenen avantajı
  bu düşük çözünürlükte güçlü çıkmadı.

### 15.3 Neden Flat AR Winner Değil?

Çünkü memorization audit kötü:

- `3162 / 4096` train exact match
- `365 / 4096` val exact match
- `333 / 4096` test exact match

Bu model metriklerde iyi görünüyor ama üretim davranışı temiz değil. Raporlarda
"memorizing baseline" olarak verilmeli.

### 15.4 Neden Deterministic Tokenizer Ana Yol Oldu?

Çünkü:

- Palette consistency garanti.
- Crisp pixel art korunuyor.
- Training stabil.
- Evaluation ve rendering basit.
- VQ-VAE kalite gate'inden geçemedi.

Bu karar proposal'dan sapma ama sonuç kalitesi açısından pragmatik ve savunulabilir.

## 16. Mevcut Artifact'lar

Önemli raporlar:

- `reports/final/final_report_narrative.md`
- `reports/final/known_metrics_comparison.md`
- `reports/final/memorization_audit_summary.md`
- `reports/final/external_baseline_and_metrics_plan.md`
- `reports/final/reproducibility_commands.md`
- `reports/option_a_progress_report.md`

Önemli sample sheet'ler:

- `reports/final/final_main_var_sample_sheet.png`
- `reports/final/final_hmar_sample_sheet.png`
- `reports/final/final_patchvq_sample_sheet.png`
- `reports/final/four_way_sample_sheet.png`
- `reports/final/final_branch_comparison_sheet.png`

Önemli kod parçaları:

- `modal_train.py`
- `scripts/train_var.py`
- `scripts/train_hmar.py`
- `scripts/train_flat_baseline.py`
- `scripts/evaluate_option_a.py`
- `scripts/evaluate_image_folders.py`
- `scripts/audit_memorization.py`
- `scripts/generate_option_a_set.py`
- `scripts/inspect_generated_set.py`
- `scripts/prepare_sd_pixl_baseline.py`

## 17. Sonraki En Mantıklı Adımlar

Sıralı öneri:

1. SD-piXL smoke run çalıştır:

```bash
modal run modal_train.py --action run-sd-pixl-smoke --sd-pixl-steps 250
```

2. SD-piXL output sample sheet'i indir ve görsel olarak incele.

3. Eğer smoke iyi görünürse küçük batch dene:

```bash
modal run modal_train.py --action run-sd-pixl-batch --num-samples 4 --sd-pixl-steps 1000
```

4. Practical SDXL/LoRA + quantization baseline ekle. Bu SD-piXL'den daha hızlı
   sample üretebilir ve external comparison için daha ölçeklenebilir olabilir.

5. Eğer zaman varsa user study için küçük ama düzgün bir form/protocol hazırla.

6. 64x64'i ancak final proposal maddeleri ve external baseline'lar daha temiz
   olduktan sonra dene.

7. Final presentation deck hazırla.

## 18. Final Değerlendirme

Implementation açısından proje core olarak güçlü bir noktada:

- Ana PixelVAR çalışıyor.
- Coarse-to-fine palette-token generation çalışıyor.
- 170K sample generation yapıldı.
- HMAR ablation tamamlandı.
- Internal baselines tamamlandı.
- Memorization audit eklendi.
- Standard image metrics eklendi.
- Modal B200 reproducibility büyük ölçüde kuruldu.

Eksikler de net:

- Proposal'daki exact VQ-VAE tokenizer tamamlanmadı.
- External diffusion baselines tamamlanmadı.
- User study yok.
- 8/16/32 codebook ablation yok.
- Scale-count ablation yok.
- 64x64 yok.

Bu yüzden raporda en dürüst claim şu olmalı:

> PixelVAR'ın deterministic palette-token versiyonu, 32x32 sprite generation
> için proposal'ın ana coarse-to-fine fikrini başarıyla gerçekleştirdi. HMAR,
> learned-token ve dataset-mixing ablation'ları denendi; fakat ana real-only
> palette-token VAR en güçlü non-memorizing proposal-path sonucu olarak kaldı.
> Proposal'daki exact VQ-VAE tokenizer, external baselines ve user study ise
> zaman, altyapı ve kalite gate nedenleriyle tamamlanamadı veya future work
> olarak kaldı.

