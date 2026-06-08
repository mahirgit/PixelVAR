# PixelVAR Üç Kişilik Sunum Metni

Tarih: 2026-06-07

Bu metin, PixelVAR projesinde şu ana kadar yapılan işleri üç kişiye bölerek
anlatmak için hazırlanmıştır. Dili bilinçli olarak rapor dilinden biraz daha
konuşma diline yakındır. Amaç, sadece "ne yaptık" demek değil; sonuçların ne
anlama geldiğini, hangi sonuçların güçlü olduğunu, hangi sonuçların yanıltıcı
olabileceğini ve neleri neden tamamlayamadığımızı açık şekilde anlatmaktır.

Önerilen dağılım:

- Kişi 1: Problem, proposal hedefi, veri, preprocessing ve 32x32 kararları.
- Kişi 2: Ana PixelVAR modeli, 170K üretim, HMAR ve metriklerin teknik yorumu.
- Kişi 3: Baseline karşılaştırmaları, memorization audit, eksikler, nedenler ve
  sonraki adımlar.

## Kısa Genel Akış

Sunumun ana mesajı şu olmalı:

"Biz proposal'daki ana fikri, yani pixel-art sprite'ları doğrudan ayrık palet
tokenları üzerinden, coarse-to-fine bir VAR modeliyle üretme fikrini çalışan
bir sisteme dönüştürdük. 32x32 çözünürlükte ana PixelVAR modelini eğittik,
HMAR masked-refinement alternatifiyle karşılaştırdık, iç baselinelar kurduk,
170K sample hedefini yerine getirdik ve standart image-generation metriklerini
ekledik. Fakat bazı proposal maddeleri, özellikle exact multi-scale VQ-VAE,
external diffusion baselinelar, sistematik ablationlar, user study ve 64x64
denemeleri tamamlanmadı. Bunları tamamlamama nedenlerimiz temelde veri erişimi,
altyapı limitleri, eğitim süresi ve önce proposal'ın ana 32x32 sonucunu
sağlamlaştırma önceliğiydi."

## Kişi 1: Problem, Proposal Hedefi ve Veri Pipeline'ı

### Açılış

Ben projenin motivasyonunu, proposal'da ne hedeflediğimizi ve veri tarafında
nereye geldiğimizi anlatacağım.

Bu projede bizim ana problemimiz pixel-art karakter sprite üretimiydi. Buradaki
önemli nokta şu: Pixel art, normal fotoğraf üretiminden farklı. Normal image
generation modellerinde renklerin sürekli RGB uzayında üretilmesi kabul
edilebilir; ama pixel art'ta keskin kenarlar, sınırlı palet, şeffaf arka plan ve
sprite'a benzeyen kompakt şekil çok daha önemli. Yani sadece "güzel görünen"
bir görsel yetmiyor. Görselin gerçekten sprite gibi davranması gerekiyor.

Proposal'daki ana fikir bu yüzden PixelVAR'dı. Burada modelin RGB piksel değeri
üretmesini istemedik. Onun yerine sprite'ı ayrık tokenlara çevirdik. Yani her
piksel ya şeffaf token oluyor ya da sabit paletteki renklerden birine denk
geliyor. Bu yaklaşımın en büyük avantajı şu: model, teorik olarak palette
olmayan renkler üretemiyor. Böylece blurry, ara renkli, diffusion tarzı
yumuşamış çıktılar yerine daha keskin ve pixel-art'a uygun çıktılar alma
şansımız artıyor.

### Proposal'daki ana hedef

Proposal'da hedeflenen sistem şuydu:

İlk olarak sprite verisini 32x32 çözünürlüğe getirmek istiyorduk. Sonra bu
sprite'ları sınırlı bir renk paletine indirgemek istiyorduk. Daha sonra
1x1'den başlayıp 32x32'ye kadar giden çok ölçekli bir temsil oluşturacaktık.
Yani model önce çok kaba bir temsil üretecek, sonra giderek daha ince detayları
tamamlayacaktı. Bu mantık VAR-style next-scale generation olarak geçiyor.

Biz bu ana hattı tamamladık. Çalışan ana modelimiz şu anda 32x32 sprite
üretiyor, 16 renkli global palet ve bir şeffaf token kullanıyor, ayrıca
1x1, 2x2, 4x4, 8x8, 16x16 ve 32x32 ölçeklerini kullanıyor.

Burada toplam token sayısı:

`1 + 4 + 16 + 64 + 256 + 1024 = 1365`

Yani bir 32x32 sprite için model toplam 1365 tokenlık bir coarse-to-fine
sequence görüyor. Bu, bizim proposal'daki ana mimari fikrimize denk geliyor.

### Veri tarafında yaşanan ilk problem

Veri tarafında ilk plan, proposal'da da geçen orijinal Kaggle sprites dataset'i
üzerinden gitmekti. Bu dataset yaklaşık 170K frame hedefiyle uyumluydu. Ama
uygulamada ciddi bir problem çıktı.

Kaggle tarafında `brentspell/sprites-dataset` için dosya listeleme denediğimizde
`403 Forbidden` hatası aldık. Tarayıcıdan baktığımızda da sayfa bulunamadı.
Yani dataset pratik olarak erişilebilir değildi.

Bu yüzden projenin durmaması için en yakın ve canlı replacement dataset'i
aradık. Ana replacement olarak `TalBarami/msd_sprites` kullandık. Bu dataset,
YingzhenLi sprites ailesine yakın ve erişilebilir bir kaynak olduğu için
seçildi.

Burada önemli nokta şu: Veri kaynağını değiştirmek ideal değildi, çünkü
proposal'daki orijinal dataset birebir kullanılamadı. Ama bu değişiklik
projenin ana fikrini bozmadı. Çünkü hâlâ sprite frame'leri üzerinde çalışıyoruz,
hâlâ 32x32 pixel-art üretimi yapıyoruz ve hâlâ ayrık palet-token yaklaşımını
test ediyoruz.

### Curated dataset sonucu

Ana replacement dataset'ten elde ettiğimiz curated sprite sayısı:

- Toplam curated frame: `93,312`
- Train: `74,664`
- Validation: `9,360`
- Test: `9,288`

Bu sayı proposal'daki 170K hedefinden düşük. Ama sonrasında modelden 170K sample
üreterek synthetic expansion tarafını ayrıca gerçekleştirdik. Yani gerçek
curated veri 93K civarında kaldı, ama generation hedefinde 170K seviyesine
çıktık.

### Preprocessing ve palet

Preprocessing tarafında bütün ana deneyleri 32x32 çözünürlükte yürüttük.
Sprite'lar şeffaf arka planla işlendi. Renk tarafında 16 renkli global palette
kullandık. Buna ek olarak şeffaflık için ayrı bir token var.

Yani token uzayımız:

- Şeffaf token: `0`
- Renk tokenları: `1..16`
- Toplam: `17` token

Bu karar çok önemli, çünkü model artık "renk değeri" üretmiyor. Model sadece bu
17 seçenekten birini seçiyor. Bu yüzden palette consistency gibi metriklerde
çok yüksek sonuç almak beklenen bir şey oluyor. Bu konuyu birazdan Kişi 2 daha
detaylı yorumlayacak.

### 32x32 neden seçildi?

Bu projede bütün ana sonuçları 32x32'de aldık. Bunun sebebi sadece kolay olması
değil. Proposal'ın ana deney hedefi zaten 32x32 sprite üretimiydi. Önce
proposal'da söylediğimiz şeyi çalışan, ölçülebilir ve savunulabilir hale
getirmek istedik.

Daha büyük çözünürlükler, özellikle 64x64, daha sonra gündeme geldi. Ama 64x64'e
geçmek basit bir "boyutu iki katına çıkaralım" meselesi değil. 32x32'de final
ölçek 1024 piksel. 64x64'te final ölçek 4096 piksel oluyor. Eğer aynı pyramid
mantığını korursak token sayısı yaklaşık şöyle büyüyor:

`1 + 4 + 16 + 64 + 256 + 1024 + 4096 = 5461`

Yani sequence uzunluğu 1365'ten 5461'e çıkıyor. Bu yaklaşık 4 kat daha uzun bir
sequence demek. Transformer tarafında bu sadece 4 kat zaman anlamına gelmeyebilir;
attention ve bellek maliyeti yüzünden çok daha ağır hissedilebilir.

Bu yüzden 64x64'i hemen yapmadık. Önce proposal'daki 32x32 ana sonucu bitirmek
istedik. Sonra Lightning AI ve Modal tarafında limit, zaman ve uzun job
problemleri yaşadık. Bunlar da 64x64 denemesini ertelememize neden oldu.

Burada açık olmak lazım: 64x64 denemesi teknik olarak mümkün, ama bu proje
akışında önce yapılması en doğru şey değildi. Çünkü 32x32 ana sistem, baselinelar,
metrikler ve audit tamamlanmadan 64x64'e geçmek, daha büyük ama daha az
kontrollü bir deney olurdu.

### Kişi 1'in vurgulaması gereken sonuç

Benim bölümümde vurgulanması gereken şey şu:

Biz dataset problemi yaşadık, ama projeyi durdurmadık. Uygun bir replacement
dataset ile pipeline'ı kurduk. 32x32 preprocessing, global palette, token map ve
multi-scale representation kısmını tamamladık. Bu da proposal'ın temel
altyapısını karşılıyor.

Ayrıca larger dimension konusunu bilinçli olarak erteledik. Çünkü önce
proposal'daki ana 32x32 sistemi tamamlamak daha mantıklıydı. 64x64'e geçmek
hem daha pahalı hem de daha uzun sürecek bir sonraki aşama olarak kaldı.

## Kişi 2: Ana PixelVAR Modeli, Sonuçlar ve Metrik Yorumu

### Ana model neydi?

Ben şimdi ana modelimizi, generation sonuçlarını ve metriklerin ne anlama
geldiğini anlatacağım.

Şu anda ana sonuç olarak tuttuğumuz model:

- Model adı: `var_sprites_v0_full`
- Çözünürlük: `32x32`
- Token uzayı: şeffaf token + 16 palette color
- Ölçekler: `[1, 2, 4, 8, 16, 32]`
- Toplam token: `1365`
- En iyi sampling ayarı: `temperature=0.8`, `top_k=8`

Bu model, bizim Option A dediğimiz ana PixelVAR modeli. Yani model coarse-to-fine
şekilde önce düşük çözünürlük tokenlarını, sonra yüksek çözünürlük tokenlarını
üretiyor.

### Ana sonuçlar

Ana model için en önemli sonuçlar şunlar:

- Sprite-feature score: `0.00147`
- Palette consistency: `1.0000`
- Opaque ratio: `0.2291`
- Reference opaque ratio: `0.2354`
- Edge density: `0.1808`
- Reference edge density: `0.1811`

Bu sayıları tek tek yorumlamak gerekiyor. Çünkü bazıları gerçekten güçlü, bazıları
ise mimari seçimden dolayı beklenen sonuçlar.

### Sprite-feature score nasıl yorumlanmalı?

Sprite-feature score bizim proposal içi değerlendirme hattımızda kullandığımız
ana kalite sinyallerinden biri. Burada düşük skor daha iyi. Ana modelin skoru
`0.00147`.

Bu sayı tek başına dış dünyadaki FID gibi standart bir skor değil. Yani "bu skor
literatürde şu seviyeye denk geliyor" diyemeyiz. Ama kendi model varyantlarımız
arasında karşılaştırma yapmak için anlamlı.

Örneğin:

- Ana PixelVAR: `0.00147`
- Generated-keep training sonucu: `0.00157`
- Mixed real + generated model real-val score: `0.00755`
- Mixed real + generated + OpenGameArt real-val score: `0.00778`

Burada ana PixelVAR'ın daha düşük olması önemli. Bu bize şunu söylüyor:
Sentetik veriyle daha büyük dataset kurmak, otomatik olarak gerçek validation
dağılımına daha iyi uyum anlamına gelmedi. Hatta mixed training tarafında skor
daha kötüleşti.

Bu sonuç bizim için önemli bir uyarıydı. Daha fazla veri her zaman daha iyi
değil. Özellikle modelin kendi ürettiği synthetic veriyi tekrar eğitimde
kullanırsak, distribution biraz kendi içine kapanabiliyor. Görsel olarak fena
olmayan örnekler üretse bile gerçek validation dağılımına göre daha iyi
olmayabiliyor.

### Palette consistency 1.0000 ne demek?

Palette consistency sonucumuz `1.0000`.

Bu ilk bakışta "mükemmel" gibi görünüyor. Ama bunu doğru yorumlamak lazım.
Bu sonuç çok iyi, evet; fakat bizim model tasarımımız açısından beklenen bir
sonuç. Çünkü model continuous RGB üretmiyor. Model sadece sabit token setinden
seçim yapıyor. Decode ederken de bu tokenlar tekrar sabit palette renklerine
çevriliyor.

Yani modelin palette dışı renk üretme ihtimali normal şartlarda yok. Bu yüzden
palette consistency'nin 1.0000 gelmesi bir "mucize kalite" göstergesi değil.
Daha çok şunu kanıtlıyor:

Bizim discrete palette-token pipeline'ımız doğru çalışıyor. Model gerçekten
palette sınırları içinde kalıyor. Bu, özellikle pixel-art için değerli bir
özellik.

Bunu şöyle anlatabiliriz:

Eğer bir diffusion modeli RGB uzayında sprite üretseydi, palette consistency'yi
1.0000 yapmak çok daha zor olurdu. Çünkü model ara renkler, blur ve palette dışı
tonlar üretebilirdi. Bizim modelimizde ise bu metrik mimari tarafından büyük
ölçüde garanti altına alınmış durumda.

Dolayısıyla palette consistency sonucu "çok ekstra şaşırtıcı" değil, ama "bu
tasarımın pixel-art'a uygun çalıştığını gösteren önemli bir doğrulama" olarak
anlatılmalı.

### Opaque ratio ne anlatıyor?

Opaque ratio, sprite'ta şeffaf olmayan piksel oranını ölçüyor. Bizim ana
modelde:

- Generated opaque ratio: `0.2291`
- Reference opaque ratio: `0.2354`

Bu değerler birbirine oldukça yakın. Bu iyi bir sonuç. Çünkü modelin sprite
boyutunu ve doluluk oranını gerçek sprite'lara yakın tuttuğunu gösteriyor.

Eğer opaque ratio çok yüksek olsaydı, model fazla dolu, arka planı taşan veya
sprite olmayan bloklar üretiyor olabilirdi. Çok düşük olsaydı, model çok küçük,
eksik veya neredeyse boş karakterler üretiyor olabilirdi.

Bizim durumda generated değer referansa yakın. Bu, modelin kaba şekil ve
şeffaflık dengesini öğrendiğini gösteriyor. Ama bu da tek başına "karakterler
çok iyi" demek değil. Sadece doluluk oranı doğru demek.

### Edge density ne anlatıyor?

Edge density, görseldeki kenar yoğunluğunu ölçen bir pixel-art proxy metriği.
Bizde:

- Generated edge density: `0.1808`
- Reference edge density: `0.1811`

Bu neredeyse referansla aynı. Bu iyi bir işaret. Pixel art'ta keskin kenarlar
önemli olduğu için edge density'nin referansa yakın olması, modelin fazla blur
veya fazla noisy üretmediğini gösteriyor.

Ama burada da dikkatli olmak lazım. Edge density'nin iyi olması, görselin
semantik olarak güzel olduğu anlamına gelmez. Mesela kenar yoğunluğu doğru olup
karakter anatomisi zayıf olabilir. Bu yüzden edge density'yi kaliteyi tek başına
kanıtlayan bir skor gibi değil, pixel-art yapısal uyumu gösteren yardımcı bir
metrik gibi anlatmalıyız.

### 170K üretim hedefi

Proposal'da 170K civarı sprite üretim hedefi vardı. Bunu generation tarafında
yerine getirdik.

Üretilen 170K set için kalite gate sonucu:

- Toplam sample: `170,000`
- Keep: `161,479`
- Review: `8,500`
- Reject: `21`

Bu sayılar ilk bakışta çok iyi görünüyor. Özellikle reject sayısının sadece 21
olması dikkat çekici. Ama bunu da doğru yorumlamak lazım.

Bu gate tamamen insan gözüyle yapılmış final kalite kontrolü değil. Otomatik
kurallar ve proxy metrikler üzerinden yapılan bir ayıklama. Bu yüzden "161K
sample insan tarafından mükemmel bulundu" diyemeyiz. Daha doğru ifade şu:

Modelden 170K üretim aldık ve otomatik kalite filtrelerine göre büyük çoğunluk
kullanılabilir sınıfta kaldı. Bu, modelin tamamen dağılmadığını ve large-scale
sampling yapabildiğini gösteriyor.

Ama bu aynı zamanda bütün sample'ların publish-ready olduğu anlamına gelmiyor.
O yüzden raporda "keep/review/reject otomatik kalite gate sonucudur" diye
açık yazmak gerekiyor.

### Generated-keep training

170K üretimden sonra keep edilen `161,479` sample ile ayrı bir training deneyi
yaptık.

Bu deneyde:

- Generated-keep dataset: `161,479`
- Train: `129,183`
- Validation/Test: `16,148`
- En iyi validation loss: `0.06441`
- Validation accuracy: `0.97924`
- Generated validation feature score: `0.00157`

Bu sonuç bize modelin kendi ürettiği dağılım üzerinde öğrenilebilir ve tutarlı
bir yapı oluşturduğunu gösterdi. Ama ana gerçek validation sonucuna göre bu
modeli ana winner yapmadık. Çünkü gerçek data dağılımına göre ana PixelVAR hâlâ
daha iyi görünüyordu.

Buradan çıkan ders şu:

Generated data, dataset büyütmek için faydalı olabilir; ama gerçek veri
kalitesinin yerini otomatik olarak almıyor. Synthetic data kullanırken
memorization, distribution drift ve self-reinforcement risklerini ayrıca
kontrol etmek gerekiyor.

### Mixed real + generated training

Daha sonra real data ile generated-keep data'yı karıştırdık.

Mixed real + generated toplamı:

- Toplam: `254,791`
- En iyi full pass validation loss: `0.05028`
- Validation accuracy: `0.98388`
- Real-validation sprite-feature score: `0.00755`

Burada validation loss ve accuracy iyi görünüyor. Ama real-val feature score
ana modele göre kötüleşti.

Bu önemli bir sonuç. Çünkü bize şunu gösterdi:

Daha büyük dataset ve daha iyi görünen token-level accuracy, her zaman daha iyi
sprite generation anlamına gelmiyor. Model, synthetic veriye de iyi uyum sağlıyor
olabilir; ama gerçek validation dağılımında istediğimiz kaliteyi artırmıyor.

Bu yüzden mixed training sonucunu "başarılı ama ana sonucu geçmedi" diye
anlatmak en dürüst yaklaşım.

### OpenGameArt ekleme deneyi

OpenGameArt tarafında da ek veri denedik.

OpenGameArt curated sonuç:

- Curated frame: `4,659`
- Group sayısı: `105`
- Train: `3,493`
- Validation: `502`
- Test: `664`

Mixed real + generated + OpenGameArt toplamı:

- Toplam: `259,450`
- En iyi full pass validation loss: `0.05408`
- Validation accuracy: `0.98295`
- Real-val score: `0.00778`

Bu da ana PixelVAR'ı geçmedi. Muhtemel neden şu: OpenGameArt görsel çeşitlilik
katıyor olabilir, ama ana MSD sprites validation dağılımına birebir uymuyor.
Ek veri daha çeşitli olsa bile hedef dağılıma uymuyorsa, skor iyileşmeyebilir.

Bu yüzden OpenGameArt'i "veri artırımı açısından denendi, ama ana model kalitesini
artırmadı" diye anlatmak gerekir.

### HMAR / masked refinement sonucu

Proposal'da Option B olarak HMAR, yani masked refinement yaklaşımı da vardı.
Bunu da implement ettik.

HMAR'da mask token `17` kullandık. Model, üretilmiş token haritasını belirli
adımlarda tekrar refine ediyor. Yani ana autoregressive generation sonrası
masked refinement ile daha iyi hale getirmeyi deniyoruz.

HMAR eğitim sonucu:

- Early stop: epoch 13
- Validation loss: `0.02025`
- Validation accuracy: `0.99216`

Bu training metrikleri çok iyi görünüyor. Ama generation kalitesi için sadece
training loss'a bakmadık. Refinement step ablation yaptık.

Ablation sonucu:

- PixelVAR baseline score: `0.00147`
- HMAR 1 step, temperature 0.8, top-k 8: `0.00189`
- HMAR 2 steps: `0.00374`
- HMAR 4 steps: `0.00366`
- HMAR 8 steps: `0.00822`

Burada en iyi HMAR sonucu 1 refinement step ile geldi. Daha fazla refinement
step'i skoru kötüleştirdi.

Bu çok önemli bir gözlem. Çünkü normalde "daha fazla refinement daha iyi olur"
diye düşünebiliriz. Ama burada öyle olmadı. Muhtemelen model her refinement
adımında küçük hataları biriktiriyor veya sprite'ın doğal yapısını bozuyor.
Bir adım düzeltme faydalı olabilirken, çok adım aşırı müdahale gibi davranıyor.

Ana sprite-feature evaluator açısından PixelVAR hâlâ HMAR'dan iyi. Ama Inception
tabanlı bazı standart metriklerde HMAR ana modele çok yakın, hatta biraz daha
iyi göründü. Bu yüzden HMAR'ı başarısız değil, güçlü bir alternatif olarak
anlatmalıyız.

### Kişi 2'nin vurgulaması gereken sonuç

Benim bölümümde ana vurgu şu olmalı:

PixelVAR ana model olarak 32x32 sprite üretiminde çalışıyor ve proposal'ın ana
fikrini doğruluyor. Palette consistency 1.0000 önemli ama beklenen bir sonuç;
çünkü model zaten palette tokenları üretiyor. Asıl daha anlamlı sonuçlar opaque
ratio ve edge density'nin referansa çok yakın olması, sprite-feature score'un
ana model lehine çıkması ve HMAR gibi alternatiflerin dürüstçe karşılaştırılmış
olması.

170K üretim hedefi yerine getirildi, ama bu setin otomatik gate ile filtrelendiği
açık söylenmeli. Mixed ve OpenGameArt deneyleri ise "daha fazla veri her zaman
daha iyi değildir" sonucunu gösterdi.

## Kişi 3: Baselinelar, Memorization Audit, Eksikler ve Sonraki Adımlar

### Baseline karşılaştırmaları neden önemliydi?

Ben şimdi karşılaştırmaları, memorization audit'i, eksik kalan proposal
maddelerini ve bundan sonra ne yapılması gerektiğini anlatacağım.

Bizim modelin iyi olup olmadığını anlamak için sadece kendi skorlarımıza bakmak
yeterli değil. Bu yüzden baselinelar kurduk ve bilinen image-generation
metriklerini ekledik.

Şu ana kadar tam çalıştırdığımız iç baselinelar:

- PixelVAR main
- HMAR step=1
- Flat raster autoregressive baseline
- Flat MaskGIT baseline

External baseline tarafında artık sadece setup değil, gerçek run sonuçları da var.
SD-piXL, SSD-1B practical diffusion ve Pokemon sprite SDXL LoRA aynı 32x32
normalization protocol'ünden geçirildi. Bunu birazdan caveat'leriyle anlatacağım.

### Kullandığımız bilinen metrikler

4096 sample üzerinden Inception V3 feature space ile şu metrikleri kullandık:

- FID
- KID
- Precision
- Recall
- Density
- Coverage
- MS-SSIM
- Exact match

Burada kısa yorum yapmak gerekiyor.

FID ve KID düşük olduğunda generated distribution'ın reference distribution'a
daha yakın olduğunu söyler. Ama önemli bir caveat var: Inception V3 doğal
fotoğraflar üzerinde eğitilmiş bir model. Pixel art sprite'lar için mükemmel
bir feature extractor değil. Yani FID/KID faydalı ve bilinen metrikler, ama
pixel-art kalitesinin mutlak gerçeği değiller.

Precision, üretilen görsellerin reference manifold'a ne kadar yakın olduğunu
gösterir. Recall, reference çeşitliliğinin ne kadar kapsandığını anlatır.
Density ve coverage de benzer şekilde distribution coverage ve yoğunluk hakkında
fikir verir.

MS-SSIM ise çeşitlilik açısından yorumlanır. Çok yüksek MS-SSIM, sample'ların
birbirine fazla benzemesi anlamına gelebilir. Bu yüzden MS-SSIM'de daha düşük
değer genellikle daha iyi diversity sinyali olarak görülür.

Exact match ise memorization için çok kritik. Çünkü bir model güzel skor alıyor
olabilir; ama eğer training görsellerini birebir kopyalıyorsa, bu gerçek
generative başarı değildir.

### Known metrics sonucu

4096 sample karşılaştırmasında metrikler şöyle çıktı:

| Method | FID lower | KID lower | Precision higher | Recall higher | Density higher | Coverage higher | MS-SSIM lower | Exact match lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat raster AR | 9.3558 | 0.002975 | 0.9172 | 0.9541 | 0.7654 | 0.8518 | 0.8281 | 0.0112 |
| HMAR step=1 | 12.6555 | 0.005209 | 0.6494 | 0.9314 | 0.4136 | 0.6165 | 0.8309 | 0.0005 |
| PixelVAR main | 13.2516 | 0.006010 | 0.6406 | 0.9292 | 0.4093 | 0.6050 | 0.8323 | 0.0007 |
| Flat MaskGIT | 67.3908 | 0.061825 | 0.1545 | 0.0317 | 0.0455 | 0.0454 | 0.9426 | 0.0000 |

Bu tabloya sadece yüzeysel bakarsak Flat raster AR açık ara en iyi görünüyor.
FID, KID, precision, recall, density ve coverage metriklerinde en iyi o.

Ama burada çok önemli bir problem var: Bu sonuç temiz değil.

Flat raster AR için memorization audit yaptığımızda modelin ciddi şekilde
ezberlediğini gördük. Yani metriklerde iyi çıkmasının nedeni gerçekten daha iyi
genelleme yapması değil, eğitim verisine çok yakın hatta birebir örnekler
üretmesi.

Bu yüzden sunumda şöyle demeliyiz:

"Raw known metrics tablosunda Flat raster AR en iyi görünüyor. Ama memorization
audit sonrasında bu sonucu temiz generative başarı olarak kabul etmiyoruz."

Bu, projenin dürüstlüğü açısından çok önemli. Çünkü sadece FID tablosunu koyup
"en iyi baseline bu" demek yanıltıcı olurdu.

### Memorization audit sonucu

4096 generated sample üzerinde exact token-map karşılaştırması yaptık. Sonuçlar:

| Model | Train exact | Val exact | Test exact | Generated duplicates | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| PixelVAR main | 157 | 13 | 11 | 4 | Low exact-match rate; disclose and keep |
| HMAR step=1 | 154 | 15 | 22 | 5 | Low exact-match rate; disclose and keep |
| Flat AR | 3162 | 365 | 333 | 206 | Memorizing; disqualify as clean winner |

Burada en kritik sayı Flat AR için train exact `3162 / 4096`. Yani ürettiği
4096 sample'ın 3162 tanesi training set ile birebir eşleşiyor. Bu çok yüksek
bir oran.

Daha da önemlisi, validation ve test exact match sayıları da yüksek:

- Validation exact: `365`
- Test exact: `333`

Processed dataset'te cross-split exact duplicate yoktu. Yani validation ve test
eşleşmelerini "zaten dataset splitlerinde duplicate vardı" diye açıklayamıyoruz.
Bu, Flat AR'ın memorization davranışının ciddi olduğunu gösteriyor.

PixelVAR ve HMAR'da da exact match sıfır değil:

- PixelVAR main: train `157`, val `13`, test `11`
- HMAR step=1: train `154`, val `15`, test `22`

Bunlar tamamen problemsiz demek değil. Bu sayıları raporda disclose etmek
gerekir. Ama Flat AR ile kıyaslandığında davranış niteliksel olarak çok farklı.
PixelVAR ve HMAR düşük exact-match oranında kalıyor; Flat AR ise açık şekilde
memorize ediyor.

Bu yüzden audit-adjusted yorumumuz şu:

- Raw known metrics winner: Flat raster AR
- Audit sonrası temiz winner değil: Flat raster AR
- En güçlü non-memorizing adaylar: PixelVAR main ve HMAR step=1
- Sprite-feature evaluator'a göre ana model: PixelVAR main
- Inception metriklerinde hafif avantaj: HMAR step=1

Bu ayrımı özellikle anlatmamız gerekiyor. Çünkü bu proje için "iyi metrik"
kadar "dürüst metrik yorumu" da önemli.

### Flat MaskGIT sonucu

Flat MaskGIT baseline mevcut ayarda başarısız oldu.

FID `67.3908`, KID `0.061825`, recall ve coverage çok düşük, MS-SSIM ise yüksek.
Bu tablo modelin hem reference distribution'ı iyi yakalayamadığını hem de
diversity tarafında zayıf kaldığını gösteriyor.

Bu sonucu şöyle anlatabiliriz:

MaskGIT fikri genel olarak kötü demiyoruz. Sadece bizim mevcut flat token setup,
training ayarları ve sprite verisi üzerinde iyi çalışmadı. Daha iyi masking
schedule, daha uzun training veya farklı architecture ile tekrar denenebilir.
Ama şu anki sonuçlarda güçlü bir baseline değil.

### External baseline durumu

External baseline tarafında kullanıcı olarak özellikle "rastgele değil, en
ilişkili ve bilinen şeylerle karşılaştıralım" demiştik. Bu doğru bir istek,
çünkü sadece kendi iç modellerimizle karşılaştırma yeterli değil.

Bu yüzden SD-piXL tarafını araştırdık ve repo içinde setup hazırladık:

- SD-piXL prompt dosyası hazırlandı.
- Palette export script'i eklendi.
- External image normalization script'i eklendi.
- SD-piXL baseline preparation script'i eklendi.
- Modal action'ları eklendi.
- Normalized output ve sample sheet pipeline'ı hazırlandı.
- SD-piXL smoke ve corrected 16-image metric batch çalıştırıldı.

Sonuç: SD-piXL bizim protocol'de zayıf çıktı. Görsel olarak centered sprite
yerine daha çok noisy/tiled bloklar üretti. Bu yüzden bunu competitive bir
baseline gibi değil, serious attempted external baseline gibi sunmalıyız.

Bunun yanında iki practical diffusion-style baseline daha çalıştırıldı:

- `segmind/SSD-1B` practical diffusion, 256-image metric run
- Pokemon trainer sprite SDXL LoRA, 256-image metric run

256-image tabloda ana sonuç şöyle:

| Method | FID | KID | Precision | Recall | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| PixelVAR main | 46.4794 | 0.007480 | 0.9102 | 0.8711 | 0.8906 |
| SSD-1B practical diffusion | 158.5819 | 0.102159 | 0.0547 | 0.4922 | 0.0352 |
| Pokemon sprite LoRA | 154.0150 | 0.136589 | 0.1172 | 0.1211 | 0.0508 |

Bunu saklamamak gerekiyor. Sunumda doğru ifade şu: "External baselinelar
çalıştırıldı, ama task/protocol farkları nedeniyle dikkatli yorumlanmalı. Bu
protocol'de hiçbiri PixelVAR'ı geçmedi."

### Neden bazı proposal maddeleri tamamlanmadı?

Burada en önemli bölüm dürüstlük. Proposal'daki her şeyi tamamlamadık. Ama
neden tamamlamadığımızı net açıklayabiliyoruz.

Tamamlanmayan veya partial kalan ana maddeler:

- Exact multi-scale VQ-VAE tokenizer
- 32-color veya 8/16/32 codebook ablation sonuçları
- Number-of-scales ablation sonuçları
- PixDiff-PIG baseline
- Exact SD 1.5 LoRA + quantization baseline
- User study, n >= 20
- 64x64 veya larger dimension experiments

Burada küçük bir güncelleme var: 8-color, 32-color ve 4-scale ablation setup'ları
artık repo içinde runnable durumda. Yani config ve Modal action tarafı hazır.
Ama GPU training/evaluation run'ları henüz çalışmadığı için bunları final sonuç
tablosuna metrik olarak eklemiyoruz.

### VQ-VAE neden ana yol olmadı?

Proposal'da multi-scale VQ-VAE tokenizer hedefi vardı. Biz neural VQ-VAE
tarafını denedik, ama kalite gate'i geçmedi. Reconstructions soft/ghosted
göründü ve code usage zayıftı. Pixel-art için bu ciddi bir problem, çünkü pixel
art'ta sharp boundary ve net palette çok önemli.

Bu yüzden exact proposal VQ-VAE yolunu ana pipeline'a koymadık. Alternatif
olarak Patch-VQ denedik.

Patch-VQ sonucu:

- Token map shape: `(93,312, 16, 16)`
- Vocab: `512`
- Kullanılan code: `143 / 512`
- Full VAR validation loss: `0.10122`
- Validation accuracy: `0.96905`
- Best decoded score: `0.04689`

Patch-VQ coherent ama blocky sonuçlar verdi. Yani teknik olarak çalıştı, ama
ana PixelVAR palette-token sonucunu geçmedi. Bu yüzden ana sonuç olarak
palette-token PixelVAR'ı tuttuk.

Bu kararı şöyle savunabiliriz:

Biz proposal'daki tokenizer fikrini denedik, ama kaliteyi düşürdüğü noktada
körü körüne devam etmedik. Pixel-art için daha keskin ve güvenilir olan discrete
palette-token yolunu ana sonuç olarak seçtik.

### 64x64 neden ertelendi?

Bu özellikle sorulabilecek bir konu. Çünkü bize daha büyük boyutları deneme
fikri geldi. 64x64'e geçebilir miydik? Evet, teknik olarak geçebilirdik. Ama o
anda doğru öncelik değildi.

Nedenleri:

Birincisi, proposal'ın ana hedefi 32x32 idi. Biz önce söylediğimiz 32x32
pipeline'ı tamamlamak istedik. Dataset, tokenizer, PixelVAR, HMAR, baselines,
metrics ve memorization audit bitmeden 64x64'e geçmek, ana raporu zayıflatırdı.

İkincisi, 64x64 token sayısını ciddi artırıyor. 32x32 pyramid 1365 token.
64x64 pyramid 5461 token seviyesine geliyor. Bu yaklaşık 4 kat sequence uzunluğu
demek. Transformer eğitiminde bu maliyet pratikte çok daha büyük hissedilebilir.

Üçüncüsü, altyapı tarafında zaten sürtünme yaşadık. Lightning AI tarafında
çalıştırma ve uzun job problemleri oldu. Modal tarafına geçmeyi düşündük,
B200 kullanımı için setup yaptık, ama limit ve harcama sınırı gibi konular
çıktı. Bazı uzun işler local tarafta iki saatlik command limit'e takıldı.
Modal limitleri daha sonra artırıldı, ama bu süreç 64x64 gibi daha pahalı bir
deneyi daha riskli hale getirdi.

Dördüncüsü, 64x64 sadece training süresini artırmaz. Evaluation, sample
generation, baseline comparison ve storage tarafını da büyütür. Yani bir kez
64x64'e geçince bütün pipeline'ı tekrar çalıştırmak gerekir.

Bu yüzden 64x64'i "yapılamadı" diye değil, "ana proposal sonuçları ve
karşılaştırmalar tamamlandıktan sonra yapılması gereken next-stage experiment"
diye çerçevelemek daha doğru.

### Lightning AI ve Modal süreci

Başta Lightning AI ile çalıştırma planı vardı. Kullanıcı tarafında run edilecek,
biz repo ve komutları hazırlayacaktık. Ama pratikte bazı altyapı problemleri
oldu. Uzun training işlerinde süre, environment ve execution kontrolü zorlaştı.

Daha sonra Modal gündeme geldi. B200 kullanmak istedik. Bunun için Modal
action'ları ve dokümantasyon eklendi. Ancak Modal'da da harcama limiti ve GPU
limitleri gibi pratik sınırlar vardı. Kullanıcı limitleri artırınca bazı
işlere devam edebildik, ama bu süreç deney planını etkiledi.

Bu yüzden çok pahalı ve uzun deneyleri, özellikle 64x64 ve external diffusion
batch run gibi işleri, ana 32x32 sonuçlar tamamlanana kadar erteledik.

Bu aslında makul bir engineering kararıydı. Çünkü sınırlı compute ile önce
proposal'ın ana iddiasını kanıtlayan deneyleri bitirmek gerekiyordu.

### Sonuçlar gerçekten ne kadar iyi?

Bu soru sunumda muhtemelen gelecek: "Sonuçlar iyi mi, çok iyi mi, suspiciously
good mu, yoksa moderate mı?"

Bunu şöyle yanıtlamak en doğru olur:

PixelVAR ana sonuçları iyi ve savunulabilir. Özellikle palette consistency,
opaque ratio ve edge density sprite yapısına uyumun güçlü olduğunu gösteriyor.
Ama palette consistency 1.0000 olduğu için bunu mucize gibi sunmamalıyız; bu
tasarımın beklenen bir sonucu.

HMAR da güçlü bir alternatif. Inception metriklerinde PixelVAR'a çok yakın ve
bazı alanlarda biraz daha iyi. Ama sprite-feature evaluator ana PixelVAR'ı daha
iyi seçti.

Flat AR suspiciously good. Çünkü raw FID/KID tarafında çok iyi görünüyor, ama
memorization audit bunun temiz bir başarı olmadığını gösteriyor. Bu yüzden Flat
AR'ı "iyi baseline" değil, "memorization riskini gösteren önemli baseline" gibi
anlatmalıyız.

Flat MaskGIT zayıf. Mevcut setup'ta beklediğimiz kaliteye yaklaşmadı.

Mixed/generated data sonuçları moderate. Büyük veri ve iyi token accuracy var,
ama ana gerçek validation kalitesini artırmadı.

External baseline tarafı artık incomplete değil, ama caveat'li. SD-piXL,
SSD-1B practical diffusion ve Pokemon sprite LoRA çalıştırıldı. Doğru ifade:
"External baselinelar ile karşılaştırdık; fakat task/protocol farkları nedeniyle
sonuçları doğrudan SOTA claim gibi değil, controlled external attempts olarak
sunuyoruz."

### Önemli dikkat noktaları

Sunumda özellikle şu noktalara dikkat çekmek gerekir:

Birincisi, bizim ana iddiamız "her metrikte SOTA'yız" değil. Ana iddia:
"Ayrık palet-token kullanan coarse-to-fine PixelVAR yaklaşımı, 32x32 pixel-art
sprite üretimi için çalışan, palette-safe ve non-memorizing bir generative
pipeline sağlıyor."

İkincisi, palette consistency çok yüksek ama bu beklenen bir sonuç. Bu, modelin
kalitesinden çok representation seçiminin başarısını gösteriyor.

Üçüncüsü, Flat AR metriklerde en iyi görünse bile memorization yüzünden temiz
winner değil. Bu belki de projenin en önemli metodolojik bulgularından biri.

Dördüncüsü, daha fazla synthetic data ana modeli otomatik olarak iyileştirmedi.
Bu da önemli, çünkü 170K üretim hedefini tamamladık ama synthetic data'yı tekrar
eğitime katmak gerçek validation kalitesinde net kazanç vermedi.

Beşincisi, 64x64 ertelendi çünkü maliyet ve zaman büyüyordu. Bu kaçınma değil,
önceliklendirme kararıydı.

### Bundan sonra ne yapılmalı?

Implementation açısından en mantıklı sonraki adımlar:

1. MDIGAN'i related work olarak bırakmak; çünkü conditional paired-pose
   imputation task'ı bizim unconditional generation protokolümüzle doğrudan adil
   karşılaştırılamıyor.
2. Zaman kalırsa user study için küçük ama düzgün bir form/protocol hazırlamak.
3. Yeni eklenen 8-color, 32-color ve 4-scale ablation run'larını Modal üzerinde
   çalıştırıp metriklerini indirmek.
4. 64x64 için sadece küçük pilot run planlamak; full training'i ancak 32x32
   final comparison bittikten sonra yapmak.

### Kişi 3'ün kapanış mesajı

Kapanışta şöyle denebilir:

Genel olarak proje proposal'ın ana fikrini çalışan bir sisteme dönüştürdü.
32x32 PixelVAR modeli var, HMAR alternatifi var, baselinelar var, metrikler var,
memorization audit var ve 170K generation hedefi tamamlandı.

Ama hâlâ eksikler var. En büyük eksikler proposal'daki exact VQ-VAE/tokenizer
ablation tarafının tamamlanmaması, user study, yeni eklenen codebook/scale
ablation setup'larının GPU sonuçlarının henüz alınmamış olması ve larger
dimension deneyleri. Bunları tamamlayamama nedenimiz ise temelde compute ve
altyapı sınırlamaları, veri erişimi problemleri ve önce ana 32x32 proposal
sonucunu sağlamlaştırma önceliğiydi.

Bu yüzden final iddiayı abartmadan kurmalıyız:

"PixelVAR, 32x32 pixel-art sprite generation için palette-safe ve coarse-to-fine
çalışan bir model olarak başarılıdır. En güçlü non-memorizing adaylardan biridir.
Ancak user study ve larger-resolution experiments tamamlanmadan
daha geniş bir SOTA iddiası yapılmamalıdır."

## Üç Kişi Arasında Net Paylaşım

### Kişi 1'in sorumluluğu

Kişi 1 şu soruları cevaplamalı:

- Problem neydi?
- Pixel art neden normal image generation'dan farklı?
- Proposal'daki ana fikir neydi?
- Dataset tarafında ne problem yaşandı?
- Hangi replacement dataset kullanıldı?
- 32x32 preprocessing, palette ve tokenization nasıl kuruldu?
- 64x64 neden hemen yapılmadı?

Kişi 1'in en önemli cümlesi:

"Biz önce proposal'ın ana 32x32 hedefini sağlamlaştırdık; larger dimension
deneylerini ise token sayısı, compute maliyeti ve altyapı problemleri nedeniyle
sonraki aşamaya bıraktık."

### Kişi 2'nin sorumluluğu

Kişi 2 şu soruları cevaplamalı:

- Ana PixelVAR modeli nasıl çalışıyor?
- En iyi sampling ayarı neydi?
- Sprite-feature score ne söylüyor?
- Palette consistency 1.0000 neden önemli ama aynı zamanda beklenen bir sonuç?
- Opaque ratio ve edge density referansa ne kadar yakın?
- 170K sample üretimi ne anlama geliyor?
- Mixed/generated data neden ana modeli geçmedi?
- HMAR neden güçlü ama ana model olarak seçilmedi?

Kişi 2'nin en önemli cümlesi:

"Palette consistency'nin 1.0000 olması çok iyi görünse de bizim discrete token
tasarımımız nedeniyle beklenen bir sonuç; asıl kalite yorumu opaque ratio, edge
density, sprite-feature score ve memorization audit ile birlikte yapılmalı."

### Kişi 3'ün sorumluluğu

Kişi 3 şu soruları cevaplamalı:

- Hangi baselinelar çalıştırıldı?
- Standard metrics tablosu ne söylüyor?
- Flat AR neden raw metriklerde iyi ama temiz winner değil?
- Memorization audit neden kritik?
- External baseline tarafında ne hazır, ne eksik?
- Proposal'da neler tamamlanmadı?
- Bunlar neden tamamlanmadı?
- Final claim nasıl kurulmalı?
- Sonraki adımlar ne olmalı?

Kişi 3'ün en önemli cümlesi:

"Flat AR metriklerde en iyi görünse de memorization audit onu temiz generative
winner olmaktan çıkarıyor; bu yüzden en dürüst yorum PixelVAR ve HMAR'ın şu ana
kadar ölçülen en güçlü non-memorizing adaylar olduğudur."

## Sunumda Kullanılabilecek Kısa Kapanış

Üç kişi de kendi bölümünü anlattıktan sonra kapanış şöyle yapılabilir:

Bu projede proposal'ın en merkezi kısmı, yani 32x32 sprite'ları ayrık palet
tokenlarıyla coarse-to-fine üretme fikri hayata geçirildi. Ana PixelVAR modeli
çalışıyor, HMAR alternatifi denendi, 170K sample üretildi, baselinelar kuruldu
ve metriklerle audit yapıldı.

En güçlü tarafımız, modelin pixel-art'a uygun bir representation kullanması.
Palette consistency'nin 1.0000 olması bu representation'ın doğru çalıştığını
gösteriyor, ama bunu tek başına kalite mucizesi gibi sunmuyoruz. Opaque ratio ve
edge density'nin referansa yakın olması da modelin sprite yapısını öğrendiğini
destekliyor.

En dikkatli olmamız gereken taraf ise metrik yorumu. Flat AR, FID gibi bilinen
metriklerde en iyi görünse bile memorization audit nedeniyle temiz winner değil.
Bu yüzden sonuçları sadece tablo üzerinden değil, audit ile birlikte yorumlamak
gerekiyor.

Eksik kalan taraflar açık: user study ve 64x64 denemeleri. Küçük
proposal-leftover ablationlar için 8-color, 32-color ve 4-scale setup'ları artık
runnable durumda, ama GPU sonuçları henüz yok. MDIGAN incelendi ama ana numeric
baseline yapılmadı; çünkü aynı karakterin başka pose'larını input olarak isteyen
conditional imputation protokolü PixelVAR'ın unconditional generation protokolüyle
doğrudan adil karşılaştırılamıyor. External baselinelar artık aynı evaluator ile
tabloya eklendi; bundan sonraki en mantıklı adım yeni stretch ablation run'larını
çalıştırıp sonuçları final tabloya eklemek.
