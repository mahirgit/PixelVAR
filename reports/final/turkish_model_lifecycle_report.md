# PixelVAR Model-by-Model Süreç Raporu

Tarih: 2026-06-07

Bu belge, projede bizim yazdığımız her model veya model-benzeri pipeline için
baştan sona ne yaptığımızı, neden yaptığımızı, ne sonuç aldığımızı ve finalde
hangi kararı verdiğimizi anlatır.

Önemli ayrım:

- **Model mimarisi**: PyTorch içinde gerçekten ayrı model sınıfı olarak yazılan
  yapı.
- **Training varyantı**: Aynı modelin farklı veriyle eğitilmiş hali.
- **Tokenizer / preprocessing pipeline**: Modelin kullandığı tokenları üreten
  sistem. Bazıları neural model değildir.
- **External baseline**: Bizim modelimiz değildir; sadece karşılaştırma için
  hazırlanan dış sistemdir.

## Kısa Özet Tablo

| Sıra | İsim | Tür | Ne oldu? | Final karar |
| ---: | --- | --- | --- | --- |
| 1 | Deterministic palette-token tokenizer | Tokenizer / preprocessing | 32x32 sprite'ları şeffaf token + 16 renk tokenına çevirdi | Ana pipeline'ın temeli |
| 2 | PixelVAR / `VARTransformer` | Ana model | Coarse-to-fine 32x32 sprite generation çalıştı | **Ana winner** |
| 3 | Generated-keep VAR | Aynı VAR, farklı veri | 170K üretimden keep edilen synthetic veriyle eğitildi | Faydalı ablation, ana model değil |
| 4 | Real + generated mixed VAR | Aynı VAR, karışık veri | Gerçek + generated veriyle eğitildi | Ana modeli geçmedi |
| 5 | OpenGameArt-mixed VAR | Aynı VAR, ekstra public veri | OGA eklendi ama real-val kalite artmadı | Ana modeli geçmedi |
| 6 | Neural `VQVAE` | Learned tokenizer modeli | Denendi, reconstruction kalitesi pixel-art için zayıf kaldı | Ana yola alınmadı |
| 7 | Patch-VQ tokenizer + Patch-VQ VAR | KMeans tokenizer + VAR | Learned-token alternatifi çalıştı ama blocky kaldı | Teknik başarı, ana model değil |
| 8 | `HMARTransformer` | Masked refinement model | Option B olarak çalıştı, çok yakın sonuç verdi | Güçlü ablation, ana winner değil |
| 9 | `FlatARTransformer` | Raster AR baseline | Raw metriklerde en iyi göründü ama ezberledi | Memorizing baseline |
| 10 | `FlatMaskGITTransformer` | Flat masked baseline | Mevcut setup'ta zayıf kaldı | Başarısız baseline |

## 1. Deterministic Palette-Token Tokenizer

Bu teknik olarak neural model değil, ama bütün ana modellerin temelini oluşturan
en önemli pipeline parçasıydı.

İlk problemimiz şuydu: pixel-art sprite üretirken continuous RGB üretmek
istemiyorduk. RGB uzayında üretim, özellikle diffusion tarzı modellerde, ara
renkler, blur ve palette dışı tonlar üretebilir. Pixel art için bu kötü bir
özellik. Bizim istediğimiz şey, modelin sadece belirli renk tokenları arasından
seçim yapmasıydı.

Bu yüzden deterministic palette-token yolunu kurduk:

- Sprite'lar 32x32'e getirildi.
- Şeffaf arka plan ayrı token olarak tutuldu.
- 16 renkli global palette çıkarıldı.
- Her piksel ya şeffaf token `0`, ya da renk tokenlarından biri oldu.
- Token aralığı `0..16` oldu.

Bu tokenizer'ın sonucu:

- Şeffaf token: `0`
- Renk tokenları: `1..16`
- Toplam vocab: `17`

Sonra bunu multi-scale hale getirdik. 32x32 final sprite için şu ölçekler
kullanıldı:

`[1, 2, 4, 8, 16, 32]`

Toplam token sayısı:

`1 + 4 + 16 + 64 + 256 + 1024 = 1365`

Bu pipeline çok kritik oldu, çünkü ana PixelVAR modelinin palette consistency
sonucunun `1.0000` gelmesini sağladı. Ama bu sonucu doğru yorumlamak gerekiyor:
Palette consistency'nin 1 olması şaşırtıcı bir mucize değil; bu temsilin doğal
sonucu. Model zaten palette dışı renk üretemiyor. Buradaki başarı, bu tasarımın
pixel-art için doğru çalışmasıdır.

Final karar:

Bu tokenizer ana pipeline olarak kaldı. En temiz, en keskin ve en kontrol
edilebilir sonuçları bu yol verdi.

## 2. PixelVAR / `VARTransformer`

Bu projenin ana modeli PixelVAR'dı. Kod tarafında ana model sınıfı
`VARTransformer`.

Amacımız şuydu: sprite'ı tek seferde düz 32x32 olarak üretmek yerine, önce kaba
ölçekleri, sonra daha ince detayları üretmek. Yani model önce 1x1, sonra 2x2,
sonra 4x4 diye gidip en sonunda 32x32 final tokenları üretiyor.

Bu yaklaşım proposal'daki Option A'ya denk geliyor.

Modelin çalışma şekli:

- Her scale için önceki daha kaba scale'ler context olarak veriliyor.
- Hedef scale'in tokenları query embedding ile tahmin ediliyor.
- Her scale içindeki tokenlar paralel tahmin ediliyor.
- Sampling sırasında model scale scale ilerliyor.

Ana model ayarları:

- Resolution: `32x32`
- Scales: `[1, 2, 4, 8, 16, 32]`
- Sequence length: `1365`
- Vocab size: `17`
- En iyi sampling: `temperature=0.8`, `top_k=8`

İlk olarak küçük debug ve overfit kontrolleri yaptık. Bu kontroller modelin
eğitim döngüsünün, tokenizer'ın, dataloader'ın ve sampling'in çalıştığını
görmek içindi. Sonra full run'a geçtik.

Ana sonuç:

- Model: `var_sprites_v0_full`
- Sprite-feature score: `0.00147`
- Palette consistency: `1.0000`
- Opaque ratio: `0.2291`
- Reference opaque ratio: `0.2354`
- Edge density: `0.1808`
- Reference edge density: `0.1811`

Bu sonuç neden iyi?

Birincisi, sprite-feature score ana karşılaştırmada en iyi sonuç oldu. İkincisi,
opaque ratio referansa yakın. Bu modelin sprite doluluğunu doğru öğrendiğini
gösteriyor. Üçüncüsü, edge density de referansa çok yakın. Bu da modelin pixel
art için gerekli keskinlik ve yapı sinyalini yakaladığını gösteriyor.

Palette consistency `1.0000` ise iyi ama beklenen bir sonuç. Çünkü model
continuous RGB üretmiyor; sadece palette tokenları üretiyor.

Final karar:

`var_sprites_v0_full` ana model ve final winner olarak kaldı.

## 3. Generated-Keep VAR

PixelVAR ana modelini eğittikten sonra 170K generation hedefi vardı. Bu hedefi
iki aşamada yaptık.

Önce 8192 sample ürettik:

- Total: `8,192`
- Keep: `7,782`
- Review: `410`
- Reject: `0`

Sonra 170K üretime çıktık:

- Total: `170,000`
- Keep: `161,479`
- Review: `8,500`
- Reject: `21`

Bu sonuç modelin büyük ölçekli sampling sırasında çökmediğini gösterdi. Ama bu
keep/review/reject ayrımı otomatik kalite gate ile yapıldı. Yani "161K sample
insan tarafından mükemmel bulundu" demek doğru olmaz. Doğru yorum şudur:
Otomatik kalite kurallarına göre üretilen sample'ların büyük çoğunluğu
kullanılabilir sınıfta kaldı.

Sonra bu keep edilen synthetic dataset ile aynı VAR mimarisini yeniden eğittik.
Bu yeni model ayrı bir mimari değil; yine `VARTransformer`. Farkı training datası.

Generated-keep training sonucu:

- Dataset: `161,479`
- Train: `129,183`
- Val/Test: `16,148`
- Best val loss: `0.06441`
- Val accuracy: `0.97924`
- Generated-validation feature score: `0.00157`

Bu skor güzel görünüyor, ama ana modelle birebir aynı referansa göre
karşılaştırılmıyor. Generated-keep model kendi generated validation dağılımına
göre iyi görünüyor. Bu yüzden bunu ana PixelVAR'ın üstüne koymadık.

Bu deneyden öğrendiğimiz şey:

Model kendi ürettiği dağılımı öğrenebiliyor, ama synthetic data'yı tekrar
training'e koymak gerçek validation dağılımında otomatik iyileşme sağlamıyor.

Final karar:

Generated-keep VAR faydalı bir ablation olarak kaldı. Ana model olmadı.

## 4. Real + Generated Mixed VAR

Generated-keep deneyinden sonra bir sonraki fikir şuydu: sadece generated veriyle
değil, gerçek veri + generated veri karışımıyla eğitirsek belki daha iyi sonuç
alırız.

Bu da yine ayrı bir mimari değil. Aynı `VARTransformer`, ama bu sefer training
datası karışık:

- Real MSD Sprites
- Filtered generated keep set

Toplam mixed dataset:

- Total: `254,791`
- Best full pass val loss: `0.05028`
- Val accuracy: `0.98388`
- Real-validation score: `0.00755`

Burada training metrikleri iyi görünüyor. Val loss ve accuracy düşük/yüksek
görünüm olarak başarılı. Ama asıl önemli nokta real-validation sprite-feature
score. Bu skor ana PixelVAR'ın `0.00147` skorundan çok daha kötü.

Bu yüzden şu sonucu çıkardık:

Daha fazla data ve daha iyi token-level accuracy, mutlaka daha iyi generation
kalitesi demek değil. Synthetic data modele daha fazla örnek sağladı, ama gerçek
validation dağılımına uyumu artırmadı.

Final karar:

Real + generated mixed VAR ana modeli geçmedi. Ablation olarak raporlandı.

## 5. OpenGameArt-Mixed VAR

Bir sonraki veri genişletme denemesi OpenGameArt oldu. Amaç, public ve farklı
kaynaklı sprite benzeri görsellerle çeşitliliği artırmaktı.

OpenGameArt curated veri:

- Curated frame: `4,659`
- Group: `105`
- Train: `3,493`
- Val: `502`
- Test: `664`

Sonra training datası şu hale geldi:

- Real MSD Sprites
- Generated keep set
- Curated OpenGameArt

Toplam:

- `259,450`

Sonuç:

- Best full pass val loss: `0.05408`
- Val accuracy: `0.98295`
- Real-validation score: `0.00778`

Bu da ana modeli geçmedi. Hatta real + generated mixed modelden de biraz daha
kötü real-val score verdi.

Muhtemel neden:

OpenGameArt faydalı ve public bir kaynak, ama ana MSD Sprites validation
dağılımıyla birebir aynı değil. Veri çeşitliliği eklemek her zaman hedef
dağılıma daha iyi uyum anlamına gelmiyor.

Final karar:

OpenGameArt pipeline çalıştı, ama ana kaliteyi artırmadı. Ana model olmadı.

## 6. Neural `VQVAE`

Proposal'da learned tokenizer hedefi vardı. Bunun için neural VQ-VAE yazdık.

Bu modelin amacı şuydu:

Sprite'ı doğrudan deterministic palette tokenlarına çevirmek yerine, neural
encoder ile latent representation üretmek, sonra bu latentleri discrete codebook
ile quantize etmek ve decoder ile sprite'a geri çevirmek.

Model parçaları:

- Encoder
- Vector quantizer
- Codebook
- Decoder
- Reconstruction loss
- VQ loss
- Commitment loss
- Code usage / perplexity takibi

Bu yol teorik olarak daha esnek. Eğer iyi çalışsaydı, palette-token'dan daha
zengin learned visual tokens elde edebilirdik.

Ama pratikte pixel-art için sorun çıktı:

- Reconstructions yumuşak/ghosted göründü.
- Keskin pixel sınırları yeterince iyi korunmadı.
- Code usage tatmin edici değildi.
- Pixel-art'ın en önemli özellikleri olan net kenar ve temiz palette yapısı
  zayıfladı.

Bu yüzden VQ-VAE'yi ana pipeline'a almadık. Burada önemli olan şu: Modeli yazdık
ve denedik, ama quality gate'i geçmediği için zorla ana sonuca koymadık.

Final karar:

Neural VQ-VAE learned-token hedefi partial / not done kaldı. Kod ve training
pipeline var, ama ana sonuç olarak kullanılmadı.

## 7. Patch-VQ Tokenizer + Patch-VQ VAR

VQ-VAE zayıf kalınca learned-token fikrini tamamen bırakmak yerine daha basit
ve daha keskin bir alternatif denedik: Patch-VQ.

Patch-VQ neural model değil. KMeans tabanlı bir tokenizer.

Çalışma şekli:

- 32x32 RGBA sprite 2x2 patch'lere bölünüyor.
- Her patch flatten ediliyor.
- MiniBatchKMeans ile patch codebook öğreniliyor.
- Her patch en yakın code'a atanıyor.
- Sonuçta 16x16 learned-code token map elde ediliyor.

Patch-VQ tokenizer sonucu:

- Source samples: `93,312`
- Token map shape: `(93,312, 16, 16)`
- Vocab size: `512`
- Used codes: `143 / 512`

Sonra bu Patch-VQ tokenları üzerinde yine `VARTransformer` eğittik. Yani burada
"Patch-VQ VAR" ayrı Transformer mimarisi değil; aynı VAR'ın farklı token
temsilinde eğitilmiş hali.

Patch-VQ VAR sonucu:

- Best val loss: `0.10122`
- Val accuracy: `0.96905`
- Best decoded setting: `temperature=1.0`, `top_k=16`
- Decoded feature score: `0.04689`

Görsel olarak Patch-VQ coherent çıktılar üretti, ama daha blocky kaldı. Bu
beklenebilir, çünkü temsil 2x2 patch code'larına dayanıyor. Bazı sprite'lar
form olarak doğru, ama ana palette-token model kadar temiz ve zengin değil.

Final karar:

Patch-VQ teknik olarak başarılı bir learned-token ablation oldu. Ama ana model
olmadı.

## 8. `HMARTransformer`

HMAR, proposal'daki Option B'ydi. Bunu ana PixelVAR'a alternatif olarak yazdık.

Ana fikir:

PixelVAR gibi coarse-to-fine ilerliyor, ama her scale içinde autoregressive
olmaktan çok masked refinement yapıyor. Yani hedef scale tokenlarının bazıları
maskeleniyor ve model bunları paralel tahmin ediyor.

HMAR'da ek olarak mask token kullandık:

- Palette vocab: `0..16`
- Mask token: `17`

Training sırasında model masked target tokenları tahmin etmeyi öğrendi. Sampling
sırasında her scale önce tamamen masked başlıyor, sonra model belli sayıda
refinement step ile tokenları dolduruyor.

Training sonucu:

- Full run early stopped at epoch 13
- Val loss: `0.02025`
- Val accuracy: `0.99216`

Bu training metrikleri çok iyi. Ama generation kalitesi için refinement ablation
yaptık.

Refinement sonuçları:

| Model | Steps | Temperature | Top-k | Feature score |
| --- | ---: | ---: | ---: | ---: |
| VAR baseline | - | `0.8` | `8` | `0.00147` |
| HMAR | `1` | `0.8` | `8` | `0.00189` |
| HMAR | `2` | `0.8` | `16` | `0.00374` |
| HMAR | `4` | `0.8` | `16` | `0.00366` |
| HMAR | `8` | `1.0` | none | `0.00822` |

En iyi HMAR sonucu 1 refinement step ile geldi. Daha fazla refinement step iyi
olmadı. Bu önemli bir sonuç, çünkü teoride daha fazla refinement daha iyi
görünebilir. Pratikte ise model her step'te küçük hataları biriktirebiliyor
veya sprite yapısını bozabiliyor.

Known metrics tarafında HMAR PixelVAR'a çok yakın ve bazı Inception metriklerinde
biraz daha iyi:

- HMAR FID: `12.6555`
- PixelVAR FID: `13.2516`
- HMAR precision: `0.6494`
- PixelVAR precision: `0.6406`
- HMAR recall: `0.9314`
- PixelVAR recall: `0.9292`

Ama sprite-feature evaluator ana PixelVAR'ı seçti.

Final karar:

HMAR başarılı bir model ve güçlü ablation. Ancak ana winner olarak promote
edilmedi.

## 9. `FlatARTransformer`

FlatAR, ana modelimize karşı baseline olarak yazıldı.

Ana fikir çok daha basit:

32x32 final token grid'ini al, raster sıraya çevir, sonra soldan sağa ve
yukarıdan aşağıya next-token prediction yap.

Bu model coarse-to-fine değil. Multi-scale hierarchy kullanmıyor. Sadece final
32x32 tokenları düz sırada öğreniyor.

İlk bakışta bu baseline'ın çok güçlü çıkması şaşırtıcıydı. Known metrics
tablosunda en iyi skorları verdi:

- FID: `9.3558`
- KID: `0.002975`
- Precision: `0.9172`
- Recall: `0.9541`
- Density: `0.7654`
- Coverage: `0.8518`

Bu tabloya tek başına bakarsak Flat AR winner gibi görünürdü.

Ama sonra memorization audit yaptık.

Flat AR memorization sonucu:

- Train exact: `3162 / 4096`
- Val exact: `365 / 4096`
- Test exact: `333 / 4096`
- Generated duplicates: `206`

Bu çok ciddi bir ezberleme sinyali. Ürettiği 4096 örneğin 3162 tanesi training
set ile birebir eşleşiyor. Ayrıca validation ve test exact match sayıları da
yüksek. Processed dataset'te cross-split exact duplicate olmadığı için bunu
split duplicate leakage ile açıklayamıyoruz.

Bu yüzden Flat AR için kararımız şu oldu:

Raw metriklerde en iyi, ama temiz generative winner değil. Suspiciously good.
Memorizing baseline olarak raporlanmalı.

Final karar:

Flat AR çok önemli bir baseline oldu, çünkü bize sadece FID/KID gibi metriklere
bakmanın yanıltıcı olabileceğini gösterdi. Ama ana model olarak kabul edilmedi.

## 10. `FlatMaskGITTransformer`

FlatMaskGIT, HMAR'a karşı daha basit bir masked-token baseline olarak yazıldı.

Ana fikir:

- Multi-scale hierarchy yok.
- Sadece final 32x32 token grid var.
- Tokenlar maskeleniyor.
- Model maskeli tokenları tahmin ediyor.
- Sampling iterative masked refinement ile yapılıyor.

Bu model aslında HMAR'ın hiyerarşik olmayan versiyonu gibi düşünülebilir.

Sonuçlar zayıf kaldı:

- FID: `67.3908`
- KID: `0.061825`
- Precision: `0.1545`
- Recall: `0.0317`
- Density: `0.0455`
- Coverage: `0.0454`
- MS-SSIM: `0.9426`

Bu sonuçlar modelin reference distribution'ı iyi yakalayamadığını gösteriyor.
Recall ve coverage çok düşük. MS-SSIM'in yüksek olması da çeşitlilik tarafında
problem olabileceğini düşündürüyor.

Burada dikkatli ifade etmek gerekiyor:

"MaskGIT kötü bir yöntemdir" demiyoruz. Sadece bizim mevcut flat 32x32
palette-token setup'ımızda, bu training ayarlarıyla iyi çalışmadı.

Final karar:

FlatMaskGIT başarısız baseline olarak raporlandı. Ana sonuç olmadı.

## Training Wrapper'ları Model Saymalı mıyız?

Kodda ayrıca Lightning wrapper sınıfları var:

- `LitVAR`
- `LitHMAR`
- `LitFlatAR`
- `LitFlatMaskGIT`
- `LitVQVAE`

Bunlar ayrı model mimarileri değil. Bunlar training, validation, optimizer,
logging ve checkpoint mantığını yöneten wrapper'lar. Yani sunumda model listesi
sayarken bunları ayrı model olarak saymamak daha doğru.

## Bizim Modelimiz Olmayan Şeyler

Şunlar bizim yazdığımız model değil:

### SD-piXL

SD-piXL external baseline olarak araştırıldı ve repo tarafında pipeline setup'ı
hazırlandı. Model bizim modelimiz değil; sadece external comparison için
kullanıldı. Daha sonra smoke ve corrected 16-image metric batch çalıştırıldı.
Sonuç görsel ve metrik olarak zayıf kaldı, bu yüzden büyük batch'e
büyütülmedi.

Durum:

- Setup: done
- Actual generation: done, 16-image metric batch
- Metrics table'a ekleme: done
- Karar: serious attempted external baseline, competitive değil

### SD 1.5 LoRA + Quantization

Proposal'da dış baseline olarak vardı. Exact SD 1.5 LoRA yapılmadı; pratik ve
erişilebilir alternatif olarak iki diffusion-style external baseline çalıştırıldı:

- `segmind/SSD-1B` practical diffusion, 256-image metric run
- `sWizad/pokemon-trainer-sprite-pixelart` SDXL LoRA, 256-image metric run

İkisi de aynı 32x32 normalization ve evaluator protocol'ünden geçirildi. İkisi
de bazı sprite benzeri çıktılar üretse de PixelVAR'ı geçmedi.

### PixDiff-PIG

Proposal'da dış baseline olarak vardı, ama çalıştırılmadı.

## Baştan Sona Ana Hikaye

Projede ilk sağlam karar, RGB üretmek yerine palette-token uzayında çalışmaktı.
Bu karar bütün sistemi daha pixel-art uyumlu hale getirdi. Deterministic
palette-token tokenizer ile 32x32 sprite'ları 17 tokenlık bir uzaya çevirdik ve
multi-scale pyramid oluşturduk.

Bunun üzerine ana `VARTransformer` modelini yazdık. Bu model proposal'ın Option
A kısmıydı. Debug, overfit ve full training aşamalarından geçti. En iyi sampling
ayarını `temperature=0.8`, `top_k=8` olarak seçtik. Bu model finalde ana winner
oldu.

Sonra bu ana modelle 170K sample üretim hedefini tamamladık. Otomatik quality
gate sonucunda 161,479 sample keep edildi. Bu generated data ile yeni training
deneyleri yaptık. Generated-keep, real+generated mixed ve OpenGameArt-mixed VAR
deneyleri çalıştı, ama hiçbiri gerçek validation skorunda ana PixelVAR'ı geçmedi.

Learned-token tarafında neural VQ-VAE yazdık. Ama reconstruction kalitesi
pixel-art için yeterli olmadı. Bunun yerine Patch-VQ alternatifini geliştirdik.
Patch-VQ teknik olarak çalıştı, ama daha blocky sonuç verdi ve ana model olmadı.

Proposal'ın Option B'si olan HMAR'ı yazdık. HMAR başarılıydı ve ana modele çok
yaklaştı. Fakat sprite-feature evaluator'da PixelVAR'ı geçemedi. Daha fazla
refinement step'in kaliteyi artırmadığını da gördük.

Son olarak baseline tarafını güçlendirdik. Flat AR ve Flat MaskGIT yazıldı.
External tarafta SD-piXL, SSD-1B practical diffusion ve Pokemon sprite SDXL
LoRA çalıştırıldı. Bu external modeller bizim modelimiz değil; sadece
karşılaştırma noktası olarak raporlandı.
Flat MaskGIT zayıf kaldı. Flat AR ise raw FID/KID metriklerinde çok iyi çıktı,
ama memorization audit'te açık şekilde ezberlediği görüldü. Bu yüzden temiz
winner olarak kabul edilmedi.

Finalde en dürüst sonuç şu:

PixelVAR ana modelimiz iyi ve savunulabilir. HMAR güçlü bir alternatif. Flat AR
metriklerde suspiciously good ama ezberliyor. Flat MaskGIT zayıf. VQ-VAE ve
Patch-VQ learned-token tarafında denendi ama ana sonucu geçmedi. Mixed-data
deneyleri dataset büyütmenin tek başına kaliteyi artırmadığını gösterdi.

## Final Model Durumu

| Model / Pipeline | Final durum |
| --- | --- |
| Deterministic palette-token tokenizer | Ana preprocessing/tokenizer |
| PixelVAR / VARTransformer | **Ana model, final winner** |
| Generated-keep VAR | Ablation, ana model değil |
| Real + generated mixed VAR | Ablation, ana model değil |
| OpenGameArt-mixed VAR | Ablation, ana model değil |
| VQVAE | Denendi, kalite gate'i geçmedi |
| Patch-VQ + VAR | Teknik olarak çalıştı, blocky kaldı |
| HMARTransformer | Güçlü ablation, ana modeli geçmedi |
| FlatARTransformer | Memorizing baseline |
| FlatMaskGITTransformer | Başarısız baseline |
