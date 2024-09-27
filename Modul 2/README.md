# Modul 2: Supervised Learning - Classification

## Pengenalan
Supervised learning adalah paradigma dalam pembelajaran mesin yang menggunakan data berlabel untuk melatih algoritma matematis. Tujuannya adalah agar algoritma mempelajari hubungan antara input (fitur) dengan output (target) sehingga dapat secara akurat memprediksi output untuk data input yang belum terlihat.

Klasifikasi merupakan subset dari supervised learning, yang mana tugasnya adalah mengelompokkan atau mengkategorikan data ke dalam kelas yang ditetapkan.

<img src="./assets/classification-machine-learning.jpg" alt="clustering" width="fit-content" height="fit-content">

## Algoritma
### K-Nearest Neighbor (KNN)
Salah satu cara mengklasifikasi adalah dengan melihat kelas mayoritas (terbanyak) yang mengelilingi data yang ingin kita prediksi, yang pada dasarnya adalah konsep KNN.

KNN mengambil sebanyak K (hyperparameter) tetangga terdekat, kemudian menentukan label kelas untuk data baru dengan melakukan voting mayoritas.

Untuk dua titik $X = (x_1, x_2, ..., x_n)$ dan $Y = (y_1, y_2, ..., y_n)$, jarak $d(X, Y)$ dapat didefinisikan sebagai:

**Euclidean Distance**

$d(X, Y) = \sqrt{\sum_{i=1}^n (x_i-y_i)^2}$

**Manhattan Distance** 

$d(X, Y) = \sum_{i=1}^n |x_i-y_i|$

**Minkowski Distance** (generalisasi euclidean dan manhattan) 

$d(X, Y) = (\sum_{i=1}^n |x_i-y_i|^p)^\frac{1}{p}$

dimana `p=1`manhattan, `p=2` euclidean.

<img src="./assets/knn.jpg" alt="knn" width="fit-content" height="fit-content">

Pada contoh diatas:
- **Jika K = 3**, tetangga dengan kelas A = 1 dan kelas B = 2. Sehingga data dikategorikan sebagai kelas B.
- **Jika K = 7**, tetangga dengan kelas A = 4 dan kelas B = 3. Sehingga data dikategorikan sebagai kelas A.

```
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)
```

KNN mudah di implementasi dan cukup bagus untuk data dengan batasan kelas yang jelas. Namun karena KNN menghitung jarak pada seluruh data relatif terhadap data input, algoritma ini sangat lambat untuk dataset besar atau berdimensi tinggi. Untuk Naive KNN, time complexity `O(ND)` dimana `N` adalah banyak data dan `D` adalah banyak dimensi.

Adapun kelemahan lain KNN yakni, karena data mengambil voting terbanyak, sangat besar kemungkinan voting bias/didominasi oleh kelas dengan data yang lebih banyak terutama pada imbalanced dataset.

### Naive Bayes (NB)
Tidak seperti KNN yang mengandalkan data tetangga, NB menghitung probabilitas kondisional kelas secara langsung. Hal ini berarti NB tidak bias terhadap kelas mayoritas seperti KNN.

Probabilitas kelas $C$ diberikan $X = (x_1, x_2, ..., x_n)$ adalah:

$P(C|X) = \frac{P(X|C) \times P(C)}{P(X)}$

NB mengasumsi semua fitur bersifat independen secara kondisional diberikan kelasnya, yang menyederhanakan probabilitas $P(X|C)$ terhadap produk probabilitas individu dari fitur:

$P(X|C) = P(x_1|C)\times P(x_2|C)\times ... \times P(x_n|C)$

Sehingga formula menjadi:

$P(C|X) = \frac{P(C)\times P(x_1|C)\times P(x_2|C)\times ... \times P(x_n|C)}{P(X)}$

Untuk mengklasifikasikan data baru, hitung $P(C|X)$ untuk setiap kelas $C$ dan menetapkan kelas dengan probabilitas tertinggi:

$Class = argmax_c \space P(C|X)$

Karena NB menggunakan prior probability dan feature likelihood, maka ketidakseimbangan kelas dapat dijelaskan secara alami, tanpa skew ke kelas mayoritas di dataset.

```
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X, Y)
```

Asumsi mengenai independensi fitur ini membuat komputasi efisien, tetapi mungkin tidak berkinerja baik jika fitur-fiturnya sangat berkorelasi seperti sebagian besar data di dunia nyata.

### Decision Tree (DT)
Dalam skenario di mana terdapat ketergantungan fitur dan interpretabilitas sangat penting, pendekatan yang lebih efektif adalah dengan merutekan titik-titik data melalui berbagai titik keputusan hingga mencapai terminal state.

Dalam DT, struktur hierarki ini digunakan untuk membuat prediksi dengan membagi data secara sistematis ke dalam subset berdasarkan nilai fitur tertentu. Struktur ini menyerupai tree, di mana setiap node mewakili keputusan berdasarkan fitur, dan leaf, atau titik akhir, menunjukkan hasil prediksi.

<img src="./assets/dt.jpg" alt="knn" width="fit-content" height="fit-content">

DT mengikuti serangkaian aturan untuk membagi data berdasarkan nilai fitur. Konstruksi DT melibatkan pemilihan fitur terbaik di setiap node untuk membagi data ke dalam kelas atau memprediksi variabel target. Untuk mengukur seberapa "baik" fitur yang dipilih sebagai node, sering digunakan Information Gain berdasarkan entropi.

Entropi $(H)$ mengukur impurity suatu dataset:

$H(S) = -\sum_{i=1}^c p_i \times \log_2 (p_i)$

Dimana:
- $S$ = dataset
- $c$ = banyak kelas
- $p_i$ = proporsi sampel dalam kelas $i$

dan ukuran pengurangan entropi setelah kumpulan data dipecah berdasarkan atribut $A$ atau Information Gain $(IG)$:

$IG(S, A) = H(S) - [\sum_{i=1}^k \frac{|S_v|}{|S|} \times H(S_v)]$

Dimana:
- $S$ = dataset
- $A$ = atribut yang digunakan untuk split
- $S_v$ = subset dari $S$ di mana atribut $A$ memiliki nilai $v$
- $|S_v|$ = jumlah sampel dalam subset $S_v$
- $|S|$ = total sampel dalam dataset $S$

Alternatif dari entropi yang digunakan untuk klasifikasi adalah Gini Impurity $(Gini)$:

$Gini(S) = 1 - \sum_{i=1}^c p_i^2$

Pada setiap node, DT memilih fitur dan threshold yang sesuai yang memaksimalkan Information Gain atau meminimalkan Gini impurity. Proses ini berlanjut secara rekursif hingga kriteria penghentian (misalnya, max depth, min samples per leaf) terpenuhi.

```
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)
```

### Random Forest (RF)
DT mudah untuk dibangun, digunakan, dan diinterpretasikan tetapi secara praktik tidak terlalu bagus. Mengutip dari buku "The Elements of Statistical Learning",

> Trees have one aspect that prevents them from being an ideal tool for predictive learning, namely inaccuracy. They seldom provide predictive accuracy comparable to the best that can be achieved with the data at hand.

dalam kata lain, tree model bekerja baik dengan data yang digunakan untuk pelatihan, namun tidak flexible ketika mengklasifikasikan sampel baru (overfitting). Salah satu cara mengatasi hal tersebut adalah dengan menggabungkan beberapa model, biasa disebut metode ensemble, untuk mengurangi variansi model.

<img src="./assets/rf.jpg" alt="knn" width="fit-content" height="fit-content">

RF mengkonstruksi banyak DT pada waktu pelatihan. Setiap tree dikonstruksi dengan cara mengambil sampel data secara acak dari dataset asli (bootstrap data). Untuk setiap split pada node, RF hanya mempertimbangkan subset acak fitur (bukan semua fitur) dari dataset.

Seluruh tree yang dibangun dengan sample data dan subset fitur acak tersebut masing-masing akan dilatih. Setelah semua tree dilatih, hasil prediksi dari setiap tree diagregasikan untuk menentukan prediksi akhir, proses bootstrapping data dan agregasi untuk membuat prediksi disebut bagging (bootstrap bagging). 

Satu hal yang perlu diketahui, proses pelatihan tersebut menghasilkan out-of-bag (OOB) data untuk setiap tree. OOB data adalah data yang tidak digunakan untuk melatih tree tersebut.

Setiap tree dalam forest memprediksi kelas dari data OOB-nya, dan akurasinya dihitung. Rata-rata akurasi dari semua tree, yang dikenal sebagai OOB error, memberikan gambaran tentang performa model secara keseluruhan. 

Untuk RF yang terdiri dari $T$ tree, OOB error (untuk klasifikasi) dapat dihitung sebagai:

$OOB Error = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\^{y_{i, OOB}} \not ={y_i} )$

Dimana:
- $N$ = jumlah total sampel dalam dataset
- $\mathbb{I}(condition) = \left\{\begin{matrix} 1 & if the condition is true \\ 0 & if the condition is false \end{matrix}\right.$
- $\^{y_{i, OOB}}$ = majority vote dari out-of-bag predictions untuk sample $i$
- ${y_i}$ = label sebenarnya dari sample $i$

Jika OOB error tinggi, model bisa diubah dengan cara menambah jumlah tree, menyesuaikan kedalaman tree, atau mengubah jumlah fitur yang digunakan untuk split node, hingga error OOB menurun. Dengan memanfaatkan OOB error, Random Forest secara otomatis menyesuaikan diri selama pelatihan untuk mencapai akurasi dan generalisasi yang baik tanpa memerlukan data validasi tambahan.

```
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)
```