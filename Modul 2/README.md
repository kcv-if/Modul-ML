# Modul 2: Supervised Learning - Classification

## Daftar Isi
- [Modul 2: Supervised Learning - Classification](#modul-2-supervised-learning---classification)
  - [Daftar Isi](#daftar-isi)
  - [Pengenalan](#pengenalan)
  - [Algoritma](#algoritma)
    - [K-Nearest Neighbor (KNN)](#k-nearest-neighbor-knn)
    - [Naive Bayes (NB)](#naive-bayes-nb)
    - [Decision Tree (DT)](#decision-tree-dt)
    - [Random Forest (RF)](#random-forest-rf)
  - [Machine Learning Techniques (Bonus)](#machine-learning-techniques-bonus)
    - [Cross-Validation (CV)](#cross-validation-cv)
      - [K-Fold](#k-fold)
      - [Stratified K-Fold](#stratified-k-fold)
    - [Hyperparameter Tuning](#hyperparameter-tuning)


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

**Contoh Implementasi:**
```python
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

**Contoh Implementasi:**
```python
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X, Y)
```

Asumsi mengenai independensi fitur ini membuat komputasi efisien, tetapi mungkin tidak berkinerja baik jika fitur-fiturnya sangat berkorelasi seperti sebagian besar data di dunia nyata.

### Decision Tree (DT)
Dalam skenario di mana terdapat ketergantungan fitur dan interpretabilitas sangat penting, pendekatan yang lebih efektif adalah dengan merutekan titik-titik data melalui berbagai titik keputusan hingga mencapai terminal state.

Dalam DT, struktur hierarki ini digunakan untuk membuat prediksi dengan membagi data secara sistematis ke dalam subset berdasarkan nilai fitur tertentu. Struktur ini menyerupai tree, di mana setiap node mewakili keputusan berdasarkan fitur, dan leaf, atau titik akhir, menunjukkan hasil prediksi.

<img src="./assets/dt.jpg" alt="dt" width="fit-content" height="fit-content">

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

**Contoh Implementasi:**
```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)
```

### Random Forest (RF)
DT mudah untuk dibangun, digunakan, dan diinterpretasikan tetapi secara praktik tidak terlalu bagus. Mengutip dari buku "The Elements of Statistical Learning",

> Trees have one aspect that prevents them from being an ideal tool for predictive learning, namely inaccuracy. They seldom provide predictive accuracy comparable to the best that can be achieved with the data at hand.

dalam kata lain, tree model bekerja baik dengan data yang digunakan untuk pelatihan, namun tidak flexible ketika mengklasifikasikan sampel baru (overfitting). Salah satu cara mengatasi hal tersebut adalah dengan menggabungkan beberapa model, biasa disebut metode ensemble, untuk mengurangi variansi model.

<img src="./assets/rf.jpg" alt="rf" width="fit-content" height="fit-content">

RF mengkonstruksi banyak DT pada waktu pelatihan. Setiap tree dikonstruksi dengan cara mengambil sampel data secara acak dari dataset asli (bootstrap data). Untuk setiap split pada node, RF hanya mempertimbangkan subset acak fitur (bukan semua fitur) dari dataset.

Seluruh tree yang dibangun dengan sample data dan subset fitur acak tersebut masing-masing akan dilatih. Setelah semua tree dilatih, hasil prediksi dari setiap tree diagregasikan untuk menentukan prediksi akhir, proses bootstrapping data dan agregasi untuk membuat prediksi disebut bagging (bootstrap bagging). 

Satu hal yang perlu diketahui, proses pelatihan tersebut menghasilkan out-of-bag (OOB) data untuk setiap tree. OOB data adalah data yang tidak digunakan untuk melatih tree tersebut.

Setiap tree dalam forest memprediksi kelas dari data OOB-nya, dan akurasinya dihitung. Rata-rata akurasi dari semua tree, yang dikenal sebagai OOB error, memberikan gambaran tentang performa model secara keseluruhan. 

Untuk RF yang terdiri dari $T$ tree, OOB error (untuk klasifikasi) dapat dihitung sebagai:

$OOB \space Error = \frac{1}{N} \sum_{i=1}^N I(y_{i, OOB} \not ={y_i} )$

Dimana:
- $N$ = jumlah total sampel dalam dataset
- $I(condition)$ = Fungsi Indikator, 1 jika jika label yang diprediksi $y_{i, OOB}$ tidak sama dengan label sebenarnya ${y_i}$ dan 0 sebaliknya
- $y_{i, OOB}$ = majority vote dari out-of-bag predictions untuk sample $i$
- ${y_i}$ = label sebenarnya dari sample $i$

Jika OOB error tinggi, model bisa diubah dengan cara menambah jumlah tree, menyesuaikan kedalaman tree, atau mengubah jumlah fitur yang digunakan untuk split node, hingga error OOB menurun. Dengan memanfaatkan OOB error, Random Forest secara otomatis menyesuaikan diri selama pelatihan untuk mencapai akurasi dan generalisasi yang baik tanpa memerlukan data validasi tambahan.

**Contoh Implementasi:**
```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)
```

## Machine Learning Techniques (Bonus)
### Cross-Validation (CV) 
Cross-Validation, atau out-of-sample testing, adalah teknik resampling yang digunakan untuk mengevaluasi performa model pada data yang tidak terlihat, mengurangi risiko overfitting. Teknik ini melibatkan pembagian dataset menjadi beberapa lipatan, menggunakan setiap lipatan sebagai test set dengan melatih lipatan yang lainnya pada model. Proses ini diulang beberapa kali, dengan setiap lipatan berfungsi sebagai test set satu kali.

#### K-Fold
Salah satu metode CV adalah K-Fold, di mana dataset dibagi menjadi `k` lipatan (fold) berukuran sama. Model dilatih sebanyak `k` kali, setiap kali menggunakan lipatan yang berbeda sebagai test set dan `k-1` lipatan yang tersisa sebagai train set. Performanya kemudian dirata-ratakan pada semua `k` percobaan untuk mendapatkan estimasi performa model.

<img src="./assets/kfold.jpg" alt="kfold" width="fit-content" height="fit-content">

Sebagai contoh, misalkan terdapat dataset dengan 100 data points dan kita menentukan banyak fold `k = 5`. Hal ini berarti K-Fold akan memiliki 5 lipatan, masing-masing dengan 20 titik data.

| Iterasi | Train Set | Test Set |
|---------|-----------|----------|
| 1       |Fold 2, 3, 4, 5|Fold 1|
| 2       |Fold 1, 3, 4, 5|Fold 2|
| 3       |Fold 1, 2, 4, 5|Fold 3|
| 4       |Fold 1, 2, 3, 5|Fold 4|
| 5       |Fold 1, 2, 3, 4|Fold 5|

**Contoh Implementasi:**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fold accuracy: {accuracy:.4f}")
```

#### Stratified K-Fold

Adapun varian dari K-Fold yaitu Stratified K-Fold. Kurang lebih cara kerja sama seperti K-Fold, namun, memastikan setiap lipatan memiliki persentase sampel yang hampir sama untuk setiap kelas target. Hal ini berguna khususnya saat menangani dataset imbalance, di mana beberapa kelas lebih sedikit  daripada yang lain.

<img src="./assets/skfold.jpg" alt="skfold" width="fit-content" height="fit-content">

**Contoh Implementasi:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fold accuracy: {accuracy:.4f}")
```

atau gunakan alternatif (yang lebih mudah), `cross_val_scorer`

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42) 
# atau gunakan skfold

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Fold accuracies: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")
```

### Hyperparameter Tuning
Proses mengoptimalkan parameter model pembelajaran mesin yang tidak dipelajari dari data tetapi ditetapkan sebelum proses pelatihan (oleh kita). Parameter ini, yang dikenal sebagai hyperparameter, mengendalikan perilaku algoritma pelatihan dan struktur model, seperti jumlah tetangga dalam K-Nearest Neighbors (KNN), atau depth decision tree.

Metode Umum untuk Hyperparameter Tuning:
- **Grid Search**: Metode brute force, dalam artian menguji semua kombinasi hiperparameter yang telah ditetapkan sebelumnya untuk menemukan yang terbaik. Cara ini efektif tetapi dapat menghabiskan banyak biaya komputasi.

```python
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier()

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X, y)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)
```

- **Random Search**: Mengambil sampel kombinasi hyperparameter secara acak. Cara ini dapat lebih efisien daripada pencarian grid karena menjelajahi hyperparameter space yang lebih besar dengan evaluasi yang lebih sedikit.

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11)
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42)

random_search.fit(X, y)

print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)
```

- **Optimasi Bayesian**: Menggunakan model probabilistik untuk menemukan hyperparameter yang optimal, menyeimbangkan eksplorasi dan eksploitasi, yang dapat lebih efisien daripada pencarian acak atau grid. Salah satu algoritma yang menggunakan model ini adalah TPE (Tree Parzen Optimizer).

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        random_state=42
    )
    
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best parameters found: ", study.best_params)
print("Best cross-validation score: ", study.best_value)
```