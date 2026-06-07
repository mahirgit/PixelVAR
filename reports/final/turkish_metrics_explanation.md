# PixelVAR Metrik Sonuçları Açıklaması

Tarih: 2026-06-07

Bu belge, PixelVAR projesinde kullandığımız metriklerin ne anlama geldiğini ve
çıkan sonuçların nasıl yorumlanması gerektiğini açıklar. Özellikle FID, KID,
precision/recall, density/coverage, MS-SSIM, palette consistency, opaque ratio,
edge density ve memorization audit sonuçları sırayla anlatılır.

En önemli uyarı:

Metrikler tek başına final karar vermek için yeterli değildir. Özellikle Flat AR
örneğinde gördüğümüz gibi, bir model FID gibi bilinen metriklerde çok iyi
görünüp aslında training set'i ezberliyor olabilir. Bu yüzden metrikleri
memorization audit ile birlikte yorumladık.

## 1. FID

FID, yani Fréchet Inception Distance, generated görsellerin dağılımı ile
reference görsellerin dağılımı arasındaki mesafeyi ölçer. Biz burada Inception
V3 feature space kullandık.

Kural:

- Daha düşük FID daha iyidir.

Bizim sonuçlarda FID:

- Flat raster AR: `9.3558`
- HMAR step=1: `12.6555`
- PixelVAR main: `13.2516`
- Flat MaskGIT: `67.3908`

İlk bakışta Flat raster AR açıkça en iyi görünüyor. HMAR ve PixelVAR birbirine
yakın. Flat MaskGIT ise çok kötü.

Ama burada çok önemli bir caveat var: Inception V3 doğal fotoğraflar üzerinde
eğitilmiş bir ağdır. Pixel-art sprite'lar için mükemmel bir feature extractor
değildir. Yani FID bizim için faydalı ve tanınan bir metrik, ama pixel-art
kalitesinin mutlak gerçeği değil.

Daha da önemlisi, Flat AR'ın düşük FID skoru memorization audit sonrası
şüpheli hale geldi. Çünkü model training örneklerini büyük oranda birebir
üretti. Bu yüzden FID tablosunda Flat AR en iyi görünse bile, temiz generative
winner olarak kabul edilmedi.

Yorum:

FID'e göre raw winner Flat AR. Audit sonrası en güvenilir adaylar PixelVAR ve
HMAR.

## 2. KID

KID, yani Kernel Inception Distance, generated ve reference dağılımları
arasındaki farkı başka bir istatistiksel ölçümle verir. FID gibi Inception
feature space kullanır.

Kural:

- Daha düşük KID daha iyidir.

Bizim sonuçlarda KID:

- Flat raster AR: `0.002975`
- HMAR step=1: `0.005209`
- PixelVAR main: `0.006010`
- Flat MaskGIT: `0.061825`

KID de FID ile aynı hikayeyi anlatıyor. Flat AR en iyi, HMAR ve PixelVAR yakın,
Flat MaskGIT kötü.

Ama yine aynı dikkat noktası geçerli: Flat AR'ın iyi KID skoru da memorization
ile açıklanabilir. Eğer model dataset örneklerini birebir kopyalıyorsa,
reference distribution'a yakın görünmesi normaldir.

Yorum:

KID, HMAR'ın PixelVAR'a biraz daha yakın hatta hafif önde olduğunu gösteriyor.
Ama Flat AR'ın KID avantajı temiz bir generative başarı olarak sunulmamalı.

## 3. Precision

Precision, generated sample'ların reference data manifold'una ne kadar yakın
olduğunu ölçer.

Kural:

- Daha yüksek precision daha iyidir.

Bizim sonuçlarda precision:

- Flat raster AR: `0.9172`
- HMAR step=1: `0.6494`
- PixelVAR main: `0.6406`
- Flat MaskGIT: `0.1545`

Flat AR burada da çok iyi görünüyor. Normalde bu, "üretilen sample'lar reference
dağılımına çok benziyor" anlamına gelir.

Ama memorization audit nedeniyle bu sonuca dikkatli bakmak gerekiyor. Eğer model
training örneklerini birebir üretiyorsa, precision'ın yüksek çıkması beklenir.
Bu durumda yüksek precision gerçek genelleme değil, ezberleme sinyaliyle
karışmış olabilir.

HMAR ve PixelVAR'ın precision değerleri birbirine yakın. HMAR biraz daha yüksek.
Bu, HMAR'ın Inception feature space'te reference'a biraz daha yakın çıktılar
ürettiğini düşündürebilir. Fakat sprite-feature evaluator ana PixelVAR'ı seçti.

Yorum:

Precision'da Flat AR raw winner, ama audit sonrası şüpheli. HMAR, PixelVAR'a
çok yakın ve biraz önde.

## 4. Recall

Recall, generated sample'ların reference distribution'daki çeşitliliği ne kadar
kapsadığını ölçer.

Kural:

- Daha yüksek recall daha iyidir.

Bizim sonuçlarda recall:

- Flat raster AR: `0.9541`
- HMAR step=1: `0.9314`
- PixelVAR main: `0.9292`
- Flat MaskGIT: `0.0317`

Burada HMAR ve PixelVAR oldukça iyi ve birbirine çok yakın. Flat AR yine en iyi,
ama memorization caveat burada da geçerli.

Flat MaskGIT'in recall değerinin `0.0317` olması ciddi bir problem. Bu, modelin
reference distribution çeşitliliğini neredeyse hiç kapsayamadığını gösteriyor.

Yorum:

Recall açısından PixelVAR ve HMAR iyi durumda. Flat MaskGIT bu metrikte net
başarısız.

## 5. Density

Density, generated sample'ların reference manifold etrafındaki yoğunluğunu
ölçer. Precision'a benzeyen ama daha yoğunluk temelli bir sinyaldir.

Kural:

- Daha yüksek density daha iyidir.

Bizim sonuçlarda density:

- Flat raster AR: `0.7654`
- HMAR step=1: `0.4136`
- PixelVAR main: `0.4093`
- Flat MaskGIT: `0.0455`

Flat AR raw olarak çok iyi. HMAR ve PixelVAR yine çok yakın. Flat MaskGIT düşük.

Bu metrikte de Flat AR'ın iyi çıkması memorization ile uyumlu. Dataset'e çok
yakın sample üretmek density'yi yükseltebilir.

Yorum:

Density, HMAR ve PixelVAR'ın birbirine çok yakın olduğunu destekliyor. Flat AR
yüksek ama audit nedeniyle temiz yorumlanamaz.

## 6. Coverage

Coverage, generated sample'ların reference distribution'ın ne kadarını kapsadığı
hakkında fikir verir.

Kural:

- Daha yüksek coverage daha iyidir.

Bizim sonuçlarda coverage:

- Flat raster AR: `0.8518`
- HMAR step=1: `0.6165`
- PixelVAR main: `0.6050`
- Flat MaskGIT: `0.0454`

HMAR ve PixelVAR burada da yakın. HMAR biraz önde. Flat MaskGIT yine başarısız.
Flat AR ise raw olarak en iyi ama memorization nedeniyle güvenilir değil.

Yorum:

Coverage tablosu, HMAR'ın Inception metriklerinde PixelVAR'a hafif avantajı
olduğunu gösteriyor. Ama final karar için tek başına yeterli değil.

## 7. MS-SSIM

MS-SSIM, sample'ların birbirine ne kadar benzediğini ölçmek için kullanılan bir
metrik olarak yorumlanabilir. Burada diversity sinyali olarak kullandık.

Kural:

- Daha düşük MS-SSIM genelde daha fazla çeşitlilik anlamına gelir.

Bizim sonuçlarda MS-SSIM:

- Flat raster AR: `0.8281`
- HMAR step=1: `0.8309`
- PixelVAR main: `0.8323`
- Flat MaskGIT: `0.9426`

Flat AR, HMAR ve PixelVAR birbirine yakın. Flat MaskGIT'in değeri çok yüksek.
Bu, Flat MaskGIT sample'larının birbirine fazla benzediğini ve diversity'nin
zayıf olduğunu düşündürüyor.

Yorum:

MS-SSIM açısından ana üç model yakın. Flat MaskGIT diversity tarafında sorunlu.

## 8. Palette Consistency

Palette consistency, generated görsellerin sabit palette içinde kalıp
kalmadığını ölçer.

Kural:

- Daha yüksek daha iyidir.
- `1.0000`, bütün piksellerin beklenen palette/token uzayında kaldığını gösterir.

Bizim sonuçlarda palette consistency:

- PixelVAR main: `1.0000`
- HMAR step=1: `1.0000`
- Flat AR: `1.0000`
- Flat MaskGIT: `1.0000`

Bu sonuç çok iyi, ama beklenen bir sonuç. Çünkü modeller continuous RGB
üretmiyor. Hepsi discrete palette tokenları üretiyor ve decode sırasında sabit
palette renklerine çevriliyor.

Yani palette consistency `1.0000` sonucunu "model olağanüstü renk kontrolü
öğrendi" diye değil, "tasarladığımız discrete palette-token pipeline doğru
çalışıyor" diye yorumlamak gerekir.

Yorum:

Palette consistency güçlü bir doğrulama metriği, ama modeller arasında winner
seçmeye yaramıyor. Çünkü hepsi aynı representation nedeniyle 1.0000 alıyor.

## 9. Opaque Ratio

Opaque ratio, sprite içinde şeffaf olmayan piksel oranını ölçer. Pixel-art
sprite için bu önemlidir, çünkü modelin çok boş veya çok dolu üretip üretmediğini
gösterir.

Kural:

- Reference'a yakın olması iyidir.

Known-metrics run'da reference opaque ratio:

- Reference: `0.2353`

Model sonuçları:

- PixelVAR main: `0.2276`
- HMAR step=1: `0.2269`
- Flat AR: `0.2321`
- Flat MaskGIT: `0.1942`

PixelVAR, HMAR ve Flat AR reference'a yakın. Flat MaskGIT daha düşük, yani
sprite'ları daha boş/eksik üretme eğiliminde olabilir.

Ana PixelVAR'ın opaque ratio'su reference'a yakın olduğu için iyi bir yapısal
sonuçtur. Ama bu tek başına görsel kalite kanıtı değildir. Sadece doluluk
dengesinin doğruya yakın olduğunu gösterir.

Yorum:

Opaque ratio, PixelVAR'ın sprite formunu iyi yakaladığını destekliyor.

## 10. Edge Density

Edge density, sprite'taki kenar yoğunluğunu ölçer. Pixel-art için yardımcı bir
metrik olarak önemlidir, çünkü pixel art'ta kenarların keskin ve yapısal olması
beklenir.

Kural:

- Reference'a yakın olması iyidir.

Known-metrics run'da reference edge density:

- Reference: `0.1807`

Model sonuçları:

- PixelVAR main: `0.1802`
- HMAR step=1: `0.1809`
- Flat AR: `0.1790`
- Flat MaskGIT: `0.1631`

PixelVAR ve HMAR reference'a çok yakın. Flat AR da yakın. Flat MaskGIT daha
düşük, bu da çıktıların daha az kenarlı, daha yumuşak veya daha az yapısal
olabileceğini düşündürüyor.

Yorum:

Edge density, PixelVAR ve HMAR'ın pixel-art yapısını iyi koruduğunu destekliyor.
Ama yine semantik kaliteyi tek başına kanıtlamaz.

## 11. Pixel Exact Match Rate

Pixel exact match rate, generated sample'ların reference sample'larla tam
eşleşme oranını gösterir. Bu değer düşük olmalıdır.

Known metrics tablosunda:

- PixelVAR main: `0.000732`
- HMAR step=1: `0.000488`
- Flat AR: `0.011230`
- Flat MaskGIT: `0.000000`

Bu metrik Flat AR için uyarı veriyor. `0.011230` oranı diğerlerinden yüksek.
Ama asıl güçlü kanıt memorization audit'ten geldi.

Yorum:

Exact match rate tek başına yeterli değil, ama Flat AR'ın şüpheli olduğunu
gösteren ilk sinyallerden biri.

## 12. Memorization Audit

Memorization audit, bu projede en kritik değerlendirmelerden biri oldu. Çünkü
raw FID/KID tablosu Flat AR'ı en iyi gösteriyordu. Eğer audit yapmasaydık,
yanlışlıkla ezberleyen modeli winner gibi sunabilirdik.

Audit 4096 generated sample üzerinde yapıldı. Generated token maps, processed
train/val/test splitleriyle exact karşılaştırıldı.

Sonuç:

| Model | Train exact | Val exact | Test exact | Generated duplicates | Yorum |
| --- | ---: | ---: | ---: | ---: | --- |
| PixelVAR main | `157` | `13` | `11` | `4` | Düşük ama sıfır değil |
| HMAR step=1 | `154` | `15` | `22` | `5` | Düşük ama sıfır değil |
| Flat AR | `3162` | `365` | `333` | `206` | Memorizing |

Flat AR'ın `3162 / 4096` train exact match üretmesi çok ciddi bir sonuç. Model
ürettiği örneklerin büyük çoğunluğunda training set'i birebir kopyalıyor.

Validation ve test exact match sayıları da yüksek. Processed dataset'te
cross-split exact duplicate olmadığı için bu eşleşmeleri split duplicate leakage
ile açıklayamıyoruz.

PixelVAR ve HMAR'da exact match sıfır değil. Bu yüzden bu sayılar raporda
disclose edilmeli. Ama Flat AR ile kıyaslandığında davranış niteliksel olarak
çok farklı.

Yorum:

Audit sonrası Flat AR temiz winner olmaktan çıkarıldı. PixelVAR ve HMAR en
güçlü non-memorizing adaylar olarak kaldı.

## 13. Final Known-Metrics Tablosu

Bu tablo 4096 generated image üzerinden, aynı validation reference folder'a karşı
hesaplandı.

| Method | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ | MS-SSIM ↓ | Exact match ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat raster AR | `9.3558` | `0.002975` | `0.9172` | `0.9541` | `0.7654` | `0.8518` | `0.8281` | `0.0112` |
| HMAR step=1 | `12.6555` | `0.005209` | `0.6494` | `0.9314` | `0.4136` | `0.6165` | `0.8309` | `0.0005` |
| PixelVAR main | `13.2516` | `0.006010` | `0.6406` | `0.9292` | `0.4093` | `0.6050` | `0.8323` | `0.0007` |
| Flat MaskGIT | `67.3908` | `0.061825` | `0.1545` | `0.0317` | `0.0455` | `0.0454` | `0.9426` | `0.0000` |

Raw metriklere göre:

- Flat AR en iyi görünüyor.
- HMAR, PixelVAR'dan Inception metriklerinde biraz önde.
- PixelVAR, HMAR'a çok yakın.
- Flat MaskGIT açık şekilde zayıf.

Audit-adjusted yoruma göre:

- Flat AR temiz winner değil, çünkü ezberliyor.
- HMAR ve PixelVAR en güçlü non-memorizing adaylar.
- PixelVAR sprite-feature evaluator'da ana winner.
- HMAR known metrics tarafında hafif avantajlı.

## 14. Final Yorum

Metrik sonuçlarını en dürüst şekilde şöyle anlatmalıyız:

Flat AR raw FID/KID/precision/recall skorlarında en iyi model gibi görünüyor.
Ancak memorization audit, bu iyi skorların temiz generative başarı olmadığını
gösterdi. Bu yüzden Flat AR'ı winner olarak değil, memorization riskini gösteren
kritik bir baseline olarak sunmalıyız.

PixelVAR ve HMAR birbirine çok yakın. HMAR, Inception tabanlı bazı metriklerde
hafif önde. PixelVAR ise proposal içi sprite-feature evaluator'da ve final model
kararında önde. Bu yüzden ana sonuç olarak PixelVAR'ı tutmak savunulabilir.

Palette consistency'nin `1.0000` olması önemli ama beklenen bir sonuçtur; çünkü
model discrete palette-token uzayında çalışıyor. Opaque ratio ve edge density
ise PixelVAR'ın sprite yapısını referansa yakın öğrendiğini destekleyen daha
anlamlı pixel-art-specific sinyallerdir.

Sonuç:

> Raw metrics alone would select Flat AR, but the audit disqualifies it as a
> clean winner. With audit-aware interpretation, PixelVAR main and HMAR step=1
> are the strongest candidates, and PixelVAR remains the main result under the
> proposal's sprite-feature evaluator.

