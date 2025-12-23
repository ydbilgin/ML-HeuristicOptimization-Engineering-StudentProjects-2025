# NeuroMech - Prediktif Bakım Sistemi

Bu proje, endüstriyel makinelerde sensör verilerini analiz ederek arıza öncesi tahmin yapabilen bir makine öğrenmesi çalışmasıdır.

## 📌 Çözülen Mühendislik Problemi

Endüstriyel üretim tesislerinde beklenmedik makine arızaları; üretim duruşlarına, yüksek bakım maliyetlerine ve iş güvenliği risklerine yol açmaktadır. Bu çalışma; sensör verilerini (sıcaklık, tork, devir hızı, takım aşınması) makine öğrenmesi ile analiz ederek arızayı **önceden tahmin etmeyi** ve planlı bakım yapılmasını hedefler.

## 🛠️ Kullanılan Yöntem ve Metodoloji

Projede dört farklı makine öğrenmesi algoritması karşılaştırılmıştır:

1. **Random Forest:**
   * Topluluk öğrenmesi yaklaşımı, 200 bağımsız karar ağacı ile oylama.

2. **XGBoost:**
   * Gradient Boosting tabanlı, sıralı hata düzeltme mekanizması.

3. **LightGBM (En İyi Model):**
   * Microsoft tarafından geliştirilen, histogram tabanlı hızlı boosting algoritması.
   * Leaf-wise büyütme stratejisi ile yüksek performans.

4. **Gradient Boosting:**
   * Temel boosting algoritması, yorumlanabilir yapı.

**Ek Teknikler:**
* **SMOTE:** Sınıf dengesizliği çözümü (28.5:1 oranı dengelendi)
* **StandardScaler:** Normalizasyon
* **Özellik Mühendisliği:** Temp_diff, Power, Wear_Torque (3 yeni özellik türetildi)

## 📊 Veri Kaynağı

* **Veri Seti:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
* **Örnek Sayısı:** 10.000 makine kaydı
* **Sensörler:** Ortam sıcaklığı, işlem sıcaklığı, devir hızı, tork, takım aşınması

## 🚀 Elde Edilen Sonuçlar

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| **LightGBM** | **%98.30** | **%82.35** | **0.7671** | **0.9856** |
| Gradient Boosting | %97.85 | %79.41 | 0.7152 | 0.9832 |
| Random Forest | %96.80 | %86.76 | 0.6484 | 0.9835 |
| XGBoost | %96.85 | %86.76 | 0.6519 | 0.9713 |

**Temel Bulgular:**
* LightGBM tüm metriklerde en iyi performansı gösterdi.
* **Tork** arıza tahmini için en kritik sensör parametresi olarak belirlendi.
* Türetilen **Power** özelliği ikinci en önemli faktör oldu.
* Sistem %98.30 doğrulukla arızaları önceden tahmin edebilmektedir.
* Endüstriyel uygulamada **%25-30 bakım maliyeti tasarrufu** potansiyeli sunmaktadır.

---

**Hazırlayan:** Ahmet BAKIR  
**Danışman:** Dr. Öğr. Üyesi Esra URAY  
**Kurum:** KTO Karatay Üniversitesi - Mekatronik Mühendisliği
