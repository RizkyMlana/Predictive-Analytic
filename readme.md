# Laporan Proyek Machine Learning - Rizky Maulana Saputra

## Domain Proyek

Energi Surya merupakan salah satu sumber energi terbarukan yang semakin banyak dimanfaatkan untuk memenuhi kebutuhan energi, terutama di sektor rumah tangga dan industri. Namun, pemanfaatan energi surya sangat bergantung pada intensitas radiasi matahari yang tidak selalu stabil dan dapat dipengaruhi oleh berbagai faktor cuaca seperti suhu, kelembapan, tekanan udara, dan waktu.

Oleh karena itu, prediksi radiasi matahari menjadi penting dalam rangka optimalisasi penggunaan panel surya, perencanaan daya pada sistem energi terbarukan, serta terbarukan, serta efisiensi distribusi energi. Dengan menggunakan pendekatan deep learning, kita dapat membangun model prediktif yang mampu mempelajari pola dari data historis cuaca untuk memperkirakan tingkat radiasi matahari di masa depan.

<!-- **Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/) -->

## Business Understanding

<!-- Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup: -->

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
1. Masih terbatasnya sistem monitoring dan prediksi radiasi matahari secara real-time
2.  Belum ada benchmark yang jelas untuk model prediksi radiasi matahari berbasis deep learning pada dataset meteorologi terbuka

### Goals

Menjelaskan tujuan dari pernyataan masalah:
1. Mengembangkan sistem prediksi radiasi matahari berbasis deep learning yang memanfaatkan data historis
2. Menyediakan baseline hasil eksperimen yang dapat digunakan oleh peneliti atau praktisi lain dalam pengembangan sistem energi surya


<!-- **Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:  -->

### Solution statementss
1. Mengembangkan model prediksi radiasi matahari menggunakan Deep Learning (Dense Layer)
Model ini dibangun dengan arsitektur feedforward neural network (dense) yang terdiri dari dari beberapa lapisan fullly connected yang bertujuan untuk mempelajari pola hubungan non-linier antara fitur fitur cuaca seperti suhu, kelembapan, tekanan udara dan lain-lainnya terhadap output berupa nilai radiasi
2. Melakukan hyperparameter tuning untuk meningkatkan performa model
dilakukan penyesuaian terhadap berbagai parameter seperti jumlah layer, jumlah neuron per layer, fungsi aktivasi, learning rate, jumlah epoch, dan batch size. Proses tuning ini bertujuan untuk memperoleh model dengan performa terbaik dalam memprediksi radiasi matahari.
3. Evaluasi model menggunakan metrik Mean Absolute Error (MAE) dan Mean Squared Error (MSE)
Metrik MAE digunakan untuk mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual. Sementara MSE memberikan penalti lebih besar terhadap error yang besar, sehingga cocok untuk mengidentifikasi prediksi yang jauh meleset. Kedua metrik ini memberikan gambaran menyeluruh terhadap performa model regresi yang dibangun.

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari Kaggle - Solar Energy, yang berisi data pengukuran kondisi cuaca dan radiasi matahari di berbagai waktu. Dataset ini digunakan untuk membangun model prediktif terhadap nilai radiasi matahari berdasarkan kondisi cuaca historis. 
[Solar Energy](https://www.kaggle.com/datasets/dronio/SolarEnergy).

### Variabel-variabel pada Solar Energy dataset adalah sebagai berikut:
Dataset memiliki sekitar 32.000 baris data yang merepresentasikan pengukuran harian atau periodik (tergantung preprocessing) dari variabel-variabel cuaca dan radiasi matahari.
- UNIXTime : Waktu dalam format UNIX timestamp.
- Data : Format waktu yang telah di-decode dari UNIXTime.
- Radiation : Nilai radiasi matahari (target yang akan diprediksi).
- Temperature : Suhu udara pada waktu pengukuran.
- Pressure : Tekanan udara.
- Humidity : Kelembaban udara.
- WindDirection(Degrees) : Arah angin dalam derajat.
- Speed : Kecepatan angin.
- TimeSunRise, TimeSunSet : Waktu matahari terbit dan terbenam.
- Time : Waktu lokal dari pengukuran.

<!-- **Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis. -->
#### Exploratory Data Analysis
Untuk memahami karakteristik data secara lebih mendalam, dilakukan beberapa tahapan eksplorasi data sebagai berikut :
1. Pemeriksaan Struktur Data dan Missing Value
Dataset terdiri dari lebih dari 32.000 baris dengan berbagai variabel cuaca seperti Temperature, Humidity, Wind Speed, Radiation sebagai target. Hasil pengecekan menunjukan bahwa tidak terdapat missing value pada dataset, sehingga tidak diperlukan penanganan khusus terdapat nilai kosong.
2. Statistik Deskriptif
Nilai rata - rata radiasi berkisar 207 W/m², dengan nilai max mendekati 1601 W/m². Ini menunjukan adanya variasi besar pada intensitas cahaya matahari sepanjang waktu pengamatan, kemungkinan dipengaruhi oleh waktu dalam sehari, musim dan kondisi cuaca
3. Distribusi Nilai Radiation
Visualisasi histogram menunjukkan bahwa distribusi Radiation cenderung right-skewed, yaitu sebagian besar nilai berada di rentang rendah (sekitar 0–200 W/m²), dan hanya sedikit yang mencapai angka sangat tinggi. Pola ini wajar karena intensitas matahari tertinggi hanya terjadi dalam waktu tertentu (misalnya, siang hari pada hari cerah).
4. Korelasi Antar Fitur
Hasil perhitungan matriks korelasi menunjukkan bahwa:
- Temperature memiliki korelasi positif yang cukup tinggi dengan Radiation (~0.63), yang masuk akal karena suhu biasanya meningkat seiring dengan meningkatnya radiasi matahari.
- Humidity menunjukkan korelasi negatif terhadap Radiation, yang berarti semakin tinggi kelembaban, semakin rendah kemungkinan intensitas cahaya matahari (karena awan atau uap air menyerap sinar).
- Fitur lain seperti Pressure, WindDirection, dan Speed menunjukkan korelasi yang lebih lemah terhadap target.
5. Insight Awal
Berdasarkan korelasi dan visualisasi, fitur Temperature dan Humidity menjadi kandidat kuat sebagai input penting untuk model prediksi. Distribusi yang tidak seimbang pada target Radiation juga perlu diperhatikan saat melakukan pemodelan karena dapat mempengaruhi performa model secara keseluruhan.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

<!-- **Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut. -->

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

<!-- _Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja. -->

