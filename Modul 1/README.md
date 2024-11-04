# Modul 1: Unsupervised Learning

## Daftar Isi
- [Definisi](#definisi)
- [Apa saja tipe unsupervised learning](#apa-saja-tipe-unsupervised-learning)
- [Clustering](#clustering-1)
    - [Prasyarat](#prasyarat)
    - [K-Means Clustering](#k-means-clustering)
    - [Hierarchical Clustering](#hierarchical-clustering)
    - [DBSCAN](#dbscan)
- [Principal Component Analysis](#principal-component-analysis)

<img src="./assets/title.webp" alt="title" width="800" height="300">

## Definisi
Model machine learning yang dilatih dengan data **tanpa** menggunakan label.

Yang dilakukan model adalah **mempelajari pola dari data**

### Apa saja tipe unsupervised learning

**Clustering**

<img src="./assets/clustering.webp" alt="clustering" width="800" height="400">

**Dimensionality Reduction**

![DR](./assets/DR.gif)

## CLUSTERING

Clustering adalah proses mengelompokkan data (objek) ke dalam kelompok-kelompok yang disebut **cluster**. Cluster dikelompokkan berdasarkan **kemiripan** antar objek.

**Kemiripan diukur dari jarak antar data**

Tujuan utama analisis cluster adalah:
- Meminimalkan Intra Cluster
- Memaksimalkan Inter-Cluster

<img src="./assets/tujuan_cluster.png" alt="tujuan_cluster" width="800" height="300">

### PRASYARAT!!!

Nah karena kita akan mencari kemiripan antar data dengan jaraknya, kita harus mengetahui cara untuk menghitung jaraknya. Ada beberapa rumus jarak yang bisa dipakai:

1. Manhattan Distance
2. Euclidean Distance
3. Minkowski Distance

![distance_1](./assets/distance_1.png)

![distance_2](./assets/distance_2.png)

### K-Means Clustering

K-Means clustering adalah algoritma yang mengelompokkan **N data** (berdasarkan fitur / atribut) ke dalam **K cluster**. Sebuah cluster di K-Means berpusat pada sebuah **titik centroid**. Selain K-Means, ada juga K-Medians dan [K-Medoids](https://esairina.medium.com/clustering-menggunakan-algoritma-k-medoids-67179a333723).

![example_1](./assets/example_1.png)

**Bagaimana K-Means Bekerja???**

![kmeans_1](./assets/kmeans_1.png)

### Hierarchical Clustering

**Definisi** = Algoritma yang menghasilkan cluster dengan cara menyusunnya seperti pohon hirarki

![HC](./assets/HC.png)

**Jenis Hierarchical Clustering**:
1. Agglomerative (Bottom-Up)
2. Divisive (Top-Down)

### DBSCAN

**Definisi** = Density-Based Spatial Clustering of Applications with Noise (DBSCAN) adalah algoritma dasar untuk pengelompokan berbasis density. DBSCAN juga bisa digunakan untuk meng-*handle* outlier.

![DBSCAN](./assets/DBSCAN.gif)

## Principal Component Analysis

**Definisi** = Teknik statistik yang digunakan untuk **mereduksi dimensi data** dengan mengubah variabel asli menjadi sekumpulan variabel baru