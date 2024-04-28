# InflationForecast
Tahmin için 8 adet ekonomik veri kullanılmaktadır.

1) 3 Aylık USD/TL Hareketli Ortalaması

2) M2 Para Arzı (1 ay gecikmeli)

3) M3 Para Arzı (1 ay gecikmeli)

4) Motorin Fiyatı

5) Toplam Kredi Hacmi(3 Aylık Hareketli Ortalama)

6) Asgari Ücret Zam Oranı (Sadece zam yapılan aylar)

7) Enflasyon Belirsizliği (TCMB Piyasa Katılımcıları Anketi Enflasyon Beklentilerinin Standart Sapması)

8) İşsizlik

9) Aylık Enflasyon(Hedef Değişken)

Tahminler için Neural Network modelleri geliştirilmiştir.

Her bir bağımsız değişkenin gelecek değerleri Prophet modeliyle tahmin edilmiş, bunlar modellere gönderilerek gelecek aylara ait enflasyon değerleri tahmin edilmiştir.Tahminler yapılırken bütün değişken kombinasyonlarıyla tahmin yapılmış,sonrasında histogramı çizdirilerek en çok tekrar eden tahminler model tahmini olarak alınmıştır.





