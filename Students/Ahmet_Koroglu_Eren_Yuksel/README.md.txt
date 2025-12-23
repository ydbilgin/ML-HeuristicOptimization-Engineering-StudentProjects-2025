# Self-Evolving Federated Learning: Genetik Algoritma ile Otonom Mimari Optimizasyonu

Bu proje, **Federe Öğrenme (Federated Learning)** ekosistemi içerisinde, derin öğrenme modellerinin mimarilerini (katman sayısı, filtre derinliği vb.) **Genetik Algoritmalar** kullanarak kendi kendine evrimleştiren (**Self-Evolving**) bir mühendislik çalışmasıdır.

##  Proje Özeti
Geleneksel derin öğrenmede model mimarisi insan müdahalesi ile tasarlanırken, bu çalışmada **Neural Architecture Search (NAS)** mantığıyla, en verimli model yapısı sistem tarafından otonom olarak belirlenir.

## Temel Teknolojiler ve Kavramlar

### 1. Self-Evolving (Kendi Kendine Evrimleşme)
Sistem, biyolojik evrimi taklit eden bir Genetik Algoritma (GA) üzerine kuruludur:
* **Genom Yapısı:** Modellerin filtre sayıları ve katman derinlikleri birer "gen" olarak tanımlanır.
* **Doğal Seçilim:** Her nesilde modeller yarıştırılır; en yüksek doğruluğu (Accuracy) veren modeller "hayatta kalır".
* **Mutasyon ve Çaprazlama:** Seçilen lider modellerin genleri üzerinde rastgele değişiklikler yapılarak bir sonraki nesle daha güçlü mimariler aktarılır.
* **Bilgi Aktarımı (Knowledge Transfer):** Ebeveyn modellerin öğrendiği ağırlıklar, ameliyat yöntemiyle çocuk modellere aktarılarak eğitimin sıfırdan başlaması engellenir.



### 2. Federated Learning (Federe Öğrenme) Simülasyonu
Veri gizliliğini korumak amacıyla veriler merkezi bir sunucuda toplanmaz:
* **Dağıtık Eğitim:** Eğitim, sanal olarak oluşturulmuş "Client A" ve "Client B" istemcileri üzerinde gerçekleştirilir.
* **Simülasyon:** MNIST ve CIFAR-10 veri setleri bu istemciler arasında paylaştırılarak gerçek dünya senaryosu taklit edilir.




## Teknik Mimari
Proje kapsamında iki farklı dinamik yapı kullanılmıştır:
* **DynamicNet:** MNIST (rakam tanıma) için evrimleşen Çok Katmanlı Algılayıcı (MLP).
* **ProDynamicCNN:** CIFAR-10 (nesne tanıma) için evrimleşen, Batch Normalization ve Global Average Pooling destekli gelişmiş Evrişimli Sinir Ağı.