import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_excel("Dataset_Sulteng_BNPB_2010_2024_195Data.xlsx")

df = df.dropna(subset=["Tahun", "Jumlah_Bencana", "Kategori_Risiko"])
df["Tahun"] = df["Tahun"].astype(int)

print("=" * 55)
print("  DATASET BENCANA SULAWESI TENGAH (BNPB 2010-2024)")
print("=" * 55)
print(f"Jumlah Data          : {len(df)}")
print(f"Jumlah Kabupaten/Kota: {df['Kabupaten_Kota'].nunique()}")
print(f"Rentang Tahun        : {df['Tahun'].min()} - {df['Tahun'].max()}")
print(f"Distribusi Kategori  :")
print(df["Kategori_Risiko"].value_counts().to_string())
print()


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
# Fitur yang digunakan:
# - Jumlah_Bencana : jumlah kejadian bencana per tahun
# - Tahun          : waktu (tren temporal)
# - Lat / Long     : posisi geografis kabupaten/kota
# - Rata_3Tahun    : rata-rata jumlah bencana 3 tahun sebelumnya
#                    (menangkap pola historis per wilayah)

df = df.sort_values(["Kabupaten_Kota", "Tahun"]).reset_index(drop=True)

df["Rata_3Tahun"] = (
    df.groupby("Kabupaten_Kota")["Jumlah_Bencana"]
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    .fillna(0)
)

FITUR = [
    "Jumlah_Bencana",
    "Tahun",
    "Latitude",
    "Longitude",
    "Rata_3Tahun"
]

TARGET = "Kategori_Risiko"

X = df[FITUR]
y = df[TARGET]


# ============================================================
# 3. ENCODING LABEL TARGET
# ============================================================
# Label Encoding mengubah kategori teks menjadi angka
# agar bisa diproses model machine learning
# Rendah=0, Sedang=1, Tinggi=2 (urutan alfabetis)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Kelas yang dikenali  : {list(le.classes_)}")
print()


# ============================================================
# 4. SCALING FITUR
# ============================================================
# StandardScaler WAJIB digunakan pada KNN karena KNN
# menghitung jarak Euclidean antar titik data.
# Tanpa scaling, fitur dengan skala besar (misal Longitude ~120)
# akan mendominasi fitur kecil (misal Jumlah_Bencana ~0-25),
# sehingga hasil prediksi menjadi bias.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 5. PEMILIHAN NILAI K TERBAIK
# ============================================================
# Mencari nilai K (jumlah tetangga) yang menghasilkan
# akurasi tertinggi menggunakan 5-Fold Cross Validation

print("=" * 55)
print("  PENCARIAN NILAI K TERBAIK (Cross Validation 5-Fold)")
print("=" * 55)
print(f"{'K':>5} {'Akurasi CV':>12} {'Std Dev':>10}")
print("-" * 30)

hasil_k = []

for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    cv_scores = cross_val_score(
        knn,
        X_scaled,
        y_encoded,
        cv=5,
        scoring="accuracy"
    )
    rata = cv_scores.mean()
    std  = cv_scores.std()
    print(f"{k:>5} {rata:>12.4f} {std:>10.4f}")
    hasil_k.append((k, rata, std))

# Pilih K dengan akurasi CV tertinggi
best_k, best_acc, best_std = max(hasil_k, key=lambda x: x[1])

print()
print(f"K Terbaik : {best_k}")
print(f"Akurasi CV: {best_acc:.4f} (+/- {best_std:.4f})")
print()


# ============================================================
# 6. TRAINING DAN EVALUASI MODEL KNN
# ============================================================
# Split data 80% training, 20% testing
# random_state=42 agar hasil dapat direproduksi

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded    # menjaga proporsi kelas di tiap split
)

knn_final = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="euclidean"
)

knn_final.fit(X_train, y_train)

y_pred = knn_final.predict(X_test)

print("=" * 55)
print("  HASIL EVALUASI MODEL KNN")
print("=" * 55)
print(f"K yang digunakan  : {best_k}")
print(f"Ukuran Data Train : {len(X_train)}")
print(f"Ukuran Data Test  : {len(X_test)}")
print()

acc = accuracy_score(y_test, y_pred)
print(f"Akurasi           : {acc:.4f} ({acc*100:.2f}%)")
print()

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    )
)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
header = f"{'':>15}" + "".join(f"{c:>10}" for c in le.classes_)
print(header)
for i, row in enumerate(cm):
    label = f"Aktual {le.classes_[i]}"
    print(f"{label:>15}" + "".join(f"{v:>10}" for v in row))
print()


# ============================================================
# 7. FUNGSI PREDIKSI PER KABUPATEN/KOTA
# ============================================================

def prediksi_kabupaten(nama_wilayah, tahun_prediksi=[2025, 2026, 2027]):
    """
    Memprediksi Kategori Risiko bencana untuk suatu wilayah
    pada tahun-tahun mendatang menggunakan model KNN yang
    sudah dilatih.

    Parameter:
        nama_wilayah    : string, nama kabupaten/kota
        tahun_prediksi  : list int, tahun yang ingin diprediksi

    Return:
        DataFrame berisi tahun dan prediksi kategori risiko
    """

    data_wil = (
        df[df["Kabupaten_Kota"] == nama_wilayah]
        .sort_values("Tahun")
    )

    if data_wil.empty:
        print(f"[PERINGATAN] Wilayah '{nama_wilayah}' tidak ditemukan.")
        return None

    # Gunakan data terakhir sebagai basis proyeksi
    terakhir = data_wil.iloc[-1].copy()

    # Ambil rata-rata bencana historis sebagai anchor
    rata_historis = data_wil["Jumlah_Bencana"].tail(3).mean()

    hasil = []
    jumlah_bencana_saat_ini = terakhir["Jumlah_Bencana"]

    for tahun in tahun_prediksi:

        # Proyeksi jumlah bencana: asumsi tren datar dengan
        # sedikit variasi (+/-1 dari rata-rata historis 3 tahun)
        jumlah_proyeksi = round(rata_historis)

        fitur_baru = {
            "Jumlah_Bencana" : jumlah_proyeksi,
            "Tahun"          : tahun,
            "Latitude"       : terakhir["Latitude"],
            "Longitude"      : terakhir["Longitude"],
            "Rata_3Tahun"    : rata_historis
        }

        X_pred = pd.DataFrame([fitur_baru])[FITUR]
        X_pred_scaled = scaler.transform(X_pred)

        pred_encoded = knn_final.predict(X_pred_scaled)[0]
        pred_label   = le.inverse_transform([pred_encoded])[0]

        # Probabilitas tiap kelas
        pred_proba = knn_final.predict_proba(X_pred_scaled)[0]
        proba_dict = {
            le.classes_[i]: round(pred_proba[i] * 100, 1)
            for i in range(len(le.classes_))
        }

        hasil.append({
            "Tahun"                 : tahun,
            "Jumlah Bencana Proyeksi": jumlah_proyeksi,
            "Prediksi Kategori"     : pred_label,
            "Prob Rendah (%)"       : proba_dict.get("Rendah", 0),
            "Prob Sedang (%)"       : proba_dict.get("Sedang", 0),
            "Prob Tinggi (%)"       : proba_dict.get("Tinggi", 0)
        })

        # Update rata historis untuk tahun berikutnya
        rata_historis = (rata_historis * 2 + jumlah_proyeksi) / 3

    return pd.DataFrame(hasil)


# ============================================================
# 8. PREDIKSI SEMUA KABUPATEN/KOTA
# ============================================================

TAHUN_PREDIKSI = [2025, 2026, 2027]

rows_output = []

print("=" * 55)
print("  PREDIKSI KATEGORI RISIKO PER KABUPATEN/KOTA")
print("=" * 55)

for wilayah in sorted(df["Kabupaten_Kota"].dropna().unique()):

    data_aktual = (
        df[df["Kabupaten_Kota"] == wilayah]
        .sort_values("Tahun")
    )

    aktual_terakhir = data_aktual.iloc[-1]

    hasil_pred = prediksi_kabupaten(wilayah, TAHUN_PREDIKSI)

    if hasil_pred is None:
        continue

    print()
    print(f"Wilayah : {wilayah}")
    print(
        f"Aktual 2024 -> Jumlah Bencana: "
        f"{int(aktual_terakhir['Jumlah_Bencana'])}, "
        f"Kategori: {aktual_terakhir['Kategori_Risiko']}"
    )

    for _, row in hasil_pred.iterrows():
        print(
            f"  Tahun {int(row['Tahun'])} -> "
            f"Prediksi: {row['Prediksi Kategori']:>7}  "
            f"| Rendah: {row['Prob Rendah (%)']:>5.1f}%  "
            f"  Sedang: {row['Prob Sedang (%)']:>5.1f}%  "
            f"  Tinggi: {row['Prob Tinggi (%)']:>5.1f}%"
        )

        rows_output.append({
            "Kabupaten_Kota"          : wilayah,
            "Tahun"                   : int(row["Tahun"]),
            "Jumlah Bencana Aktual 2024": int(aktual_terakhir["Jumlah_Bencana"]),
            "Kategori Aktual 2024"    : aktual_terakhir["Kategori_Risiko"],
            "Jumlah Bencana Proyeksi" : int(row["Jumlah Bencana Proyeksi"]),
            "Prediksi Kategori"       : row["Prediksi Kategori"],
            "Prob Rendah (%)"         : row["Prob Rendah (%)"],
            "Prob Sedang (%)"         : row["Prob Sedang (%)"],
            "Prob Tinggi (%)"         : row["Prob Tinggi (%)"]
        })


# ============================================================
# 9. SIMPAN HASIL KE EXCEL
# ============================================================

df_output = pd.DataFrame(rows_output)

df_output.to_excel(
    "hasil_prediksi_knn_bencana_sulteng.xlsx",
    index=False
)

print()
print("=" * 55)
print("File output:")
print("  hasil_prediksi_knn_bencana_sulteng.xlsx")
print("=" * 55)