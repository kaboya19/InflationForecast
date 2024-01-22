# InflationForecast
Tahmin için 17 adet ekonomik veri kullanılmaktadır.

1) 3 Aylık USD/TL Hareketli Ortalaması

2) M2 Para Arzı (1 ay gecikmeli)

3) M3 Para Arzı (1 ay gecikmeli)

4) Motorin Fiyatı

5) Politika Faizi

6) Ortalama Kredi Faizi

7) Ortalama 3 Aylık Mevduat Faizi

8) Kamu Borç Stoğu(2 Aylık Hareketli Ortalama)

9) Sanayi Üretim Endeksi

10) Perakende Satış Hacmi

11) Toplam Kredi Hacmi(3 Aylık Hareketli Ortalama)

12) Asgari Ücret Zam Oranı (Sadece zam yapılan aylar)

13) Enflasyon Belirsizliği (TCMB Piyasa Katılımcıları Anketi 12 Ay Sonrası Enflasyon Beklentilerinin Standart Sapması)

14) Reel Efektif Döviz Kuru (TÜFE Bazlı)

15) Reel Efektif Döviz Kuru (ÜFE Bazlı)

16) İşsizlik

17) Aylık Enflasyon(Hedef Değişken)

Her bir bağımsız değişkenin gelecek değerleri Prophet modeliyle tahmin edilmiş, bunlar modellere gönderilerek gelecek aylara ait enflasyon değerleri tahmin edilmiştir.Tahminler yapılırken bütün değişken kombinasyonlarıyla tahmin yapılmış,sonrasında histogramı çizdirilerek en çok tekrar eden tahminler model tahmini olarak alınmıştır.

Kullanılan Modeller:

1) Lineer Regresyon

2) Bayesian Regresyon

3) Gaussian Regresyon

4) Kernel Regresyon

5) Lasso Regresyon

6) Lars Regresyon

7) SGD Regresyon

9) Robust Regresyon

10) LSTM



