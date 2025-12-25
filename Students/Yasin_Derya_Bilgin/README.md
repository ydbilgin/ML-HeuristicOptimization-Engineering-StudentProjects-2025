# VRP - Araç Rotalama Projesi (Vehicle Routing Problem)

Merhaba, bu proje benim **Araç Rotalama Problemi (VRP)** üzerine yaptığım çalışmamdır. Python kullanarak, karmaşık lojistik problemlerini nasıl çözebileceğimizi araştırdım.

## Proje Ne İşe Yarıyor?
Bir kargo firmanız veya dağıtım ağınız olduğunu düşünün. Elinizde 5 tane araç ve gitmesi gereken 100 tane adres var. Hangi araç hangi adreslere gitmeli? Hangi sırayla gitmeli ki en az benzini yakalım ve en kısa sürede işi bitirelim? İşte bu proje tam olarak bunu hesaplıyor.

Temel olarak iki problemi çözüyoruz:
1. **Adresleri Haritaya İşleme (Geocoding):** Elimizdeki açık adresleri (Örn: "Kızılay Meydanı, Ankara") bilgisayarın anlayacağı koordinatlara (Enlem: 39.92, Boylam: 32.85) çeviriyoruz.
2. **Optimizasyon (Genetik Algoritma):** Binlerce olası rota arasından en iyisini bulmak için biyolojideki evrim teorisinden esinlenen "Genetik Algoritma" yöntemini kullanıyoruz.

## Temel Terimler

Eğer bu kelimeleri ilk defa duyuyorsanız, işte anlamları:

- **Geocode (Coğrafi Kodlama):** "Atatürk Bulvarı No:5" gibi bir yazı bilgisayar için anlamsızdır. Bilgisayar sayılarla çalışır. Geocode işlemi, bu adresi alıp haritadaki tam noktasına (Örn: 32.85 Enlem, 39.92 Boylam) çevirme işlemidir. Projede bunun için **Nominatim** kullanıyoruz (ücretsiz bir harita servisi).
- **OSRM (Open Source Routing Machine):** Haritada iki nokta arasındaki mesafeyi ölçerken iki yol vardır:
    1. **Kuş Uçuşu:** Dümdüz çizgi çekersiniz. Helikopteriniz yoksa bu gerçekçi değildir.
    2. **OSRM:** Gerçek yolları, sokakları, tek yönleri ve trafik kurallarını hesaba katar. "Buradan sola dönülmez, arka sokaktan dolanman lazım" diyerek gerçek sürüş mesafesini hesaplar.
    *(Not: Bu projede varsayılan olarak kuş uçuşu (haversine) hesaplanır, OSRM için ekstra kurulum gerekir.)*

## Kurulum ve Çalıştırma

Bu projeyi kendi bilgisayarınızda çalıştırmak için Terminal veya Komut İstemi'ni kullanacağız.

### 1. Hazırlık
Gerekli kütüphaneleri (folium, matplotlib vb.) kurmamız gerekebilir.
Not: Eğer `pip` komutu çalışmazsa Python yüklü olduğundan emin olun.

### 2. Adım: Adresleri Koordinata Çevirme (Geocoding)
Adreslerinizi `addresses` klasörü içine `.txt` dosyası olarak kaydedin. Sonra şu sihirli komutu çalıştırın:

```bash
python src/geocode_addresses.py --depots-file addresses/ankara_depo.txt --customers-file addresses/ankara_musteri.txt --prefix ankara
```
*Bu işlem internet gerektirir çünkü adresleri harita servisinden sorar.*

### 3. Adım: Rotaları Hesaplama
Koordinatlarımız hazırsa (data klasörüne geldiyse), operasyonu başlatıyoruz:

```bash
python src/run_vrp.py --depots-csv data/geocoded/ankara_depots.csv --customers-csv data/geocoded/ankara_customers.csv --vehicles 5 --pop-size 100 --generations 500 --out output/ankara_routes.json --plot-html output/ankara_harita.html
```

#### Önemli Ayarlar (Değişkenler):
Bu komutta değiştirebileceğiniz önemli sayılar şunlardır:
- **--vehicles 5**: "Elimde 5 tane araç var". Bunu elinizdeki araç sayısına göre değiştirin.
- **--generations 500**: "Algoritma 500 tur dönsün". Bu sayıyı artırırsanız program daha uzun sürer ama **daha iyi (kısa) rotalar** bulma şansı artar.
- **--pop-size 100**: "Her turda 100 farklı plan dene". Bu sayıyı artırmak da çözüm kalitesini artırır.
- **--plot-html**: Sonuçta oluşacak harita dosyasının adı.

## Çıktıların İncelenmesi
Program çalışmayı bitirdiğinde `output` klasörüne bakın:
1. **HTML Dosyası (ankara_harita.html):** Bu dosyayı çift tıklayıp tarayıcıda açın. Rotaların harita üzerinde renkli çizgilerle çizildiğini göreceksiniz. Her renk ayrı bir aracı temsil eder.
2. **JSON Dosyası:** Rotaların sayısal verilerini içerir (hangi araç hangi sırayla nereye gidiyor, toplam kaç km yol yaptı vb.).

## Klasör Yapısı
- **src/**: Projenin beyni. Bütün Python kodları burada.
- **addresses/**: Ham adres listelerimiz.
- **data/**: İşlenmiş veriler (koordinatlar).
- **Jupyter notebooks/**: Projeyi geliştirirken yaptığım denemeler ve analizler.
- **examples/**: Daha önceki çalışmalardan örnek harita çıktıları.
- **output/**: Sizin çalıştırdığınızda oluşacak sonuçlar.

Sorularınız olursa bana ulaşabilirsiniz. İyi çalışmalar!

