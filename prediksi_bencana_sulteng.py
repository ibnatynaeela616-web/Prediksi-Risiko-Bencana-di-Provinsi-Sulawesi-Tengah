import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# ─────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────
df_raw = pd.read_csv("Dataset_Sulteng_BNPB_2010_2024_195Data.csv")

# ─────────────────────────────────────────────
# Statistik Deskriptif
# ─────────────────────────────────────────────
jb = df_raw["Jumlah_Bencana"].values
stat_sw, p_sw = stats.shapiro(jb)

print("=" * 60)
print("  STATISTIK DESKRIPTIF — JUMLAH BENCANA SULTENG 2010-2024")
print("=" * 60)
print(f"  Jumlah Data  : {len(df_raw)} baris | {df_raw['Kabupaten_Kota'].nunique()} Kab/Kota")
print(f"  Mean         : {np.mean(jb):.4f} kejadian/tahun")
print(f"  Median       : {np.median(jb):.4f} kejadian/tahun")
print(f"  Std Dev      : {np.std(jb):.4f}")
print(f"  Min – Max    : {np.min(jb):.0f} – {np.max(jb):.0f} kejadian")
print(f"  Skewness     : {stats.skew(jb):.4f}")
print(f"  Kurtosis     : {stats.kurtosis(jb):.4f}")
print(f"  Shapiro-Wilk : stat={stat_sw:.4f}, p={p_sw:.4f}  →  "
      + ("Normal" if p_sw > 0.05 else "Tidak Normal"))
print()

# ─────────────────────────────────────────────
# Preprocessing (tanpa output)
# ─────────────────────────────────────────────
FITUR  = ["Jumlah_Bencana", "Tahun"]
TARGET = "Kategori_Risiko"

X = df_raw[FITUR].values
y = df_raw[TARGET].values

le = LabelEncoder()
le.fit(["Rendah", "Sedang", "Tinggi"])
y_enc = le.transform(y)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# Pencarian K Optimal (proses diam, hanya tampil hasilnya)
# ─────────────────────────────────────────────
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
k_values = range(1, 16)
acc_list = []
std_list = []

for k in k_values:
    knn    = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    scores = cross_val_score(knn, X_scaled, y_enc, cv=skf, scoring="accuracy")
    acc_list.append(scores.mean())
    std_list.append(scores.std())

best_k   = list(k_values)[np.argmax(acc_list)]
best_acc = max(acc_list)

# ─────────────────────────────────────────────
# Evaluasi Model Final
# ─────────────────────────────────────────────
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")

acc_cv  = cross_val_score(knn_final, X_scaled, y_enc, cv=skf, scoring="accuracy")
prec_cv = cross_val_score(knn_final, X_scaled, y_enc, cv=skf, scoring="precision_weighted")
rec_cv  = cross_val_score(knn_final, X_scaled, y_enc, cv=skf, scoring="recall_weighted")
f1_cv   = cross_val_score(knn_final, X_scaled, y_enc, cv=skf, scoring="f1_weighted")

print("=" * 60)
print(f"  EVALUASI MODEL KNN — Stratified 5-Fold CV")
print("=" * 60)
print(f"  K Optimal  : {best_k}")
print(f"  {'Metrik':<15} {'Rata-rata':>10}  {'Std Dev':>10}")
print("  " + "-" * 40)
print(f"  {'Accuracy':<15} {acc_cv.mean():>10.4f}  {acc_cv.std():>10.4f}")
print(f"  {'Precision':<15} {prec_cv.mean():>10.4f}  {prec_cv.std():>10.4f}")
print(f"  {'Recall':<15} {rec_cv.mean():>10.4f}  {rec_cv.std():>10.4f}")
print(f"  {'F1-Score':<15} {f1_cv.mean():>10.4f}  {f1_cv.std():>10.4f}")
print()

# ─────────────────────────────────────────────
# Training Final dan Prediksi
# ─────────────────────────────────────────────
knn_final.fit(X_scaled, y_enc)
y_pred_enc = knn_final.predict(X_scaled)
y_pred     = le.inverse_transform(y_pred_enc)

acc_full = accuracy_score(y_enc, y_pred_enc)

# Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=le.classes_)
print("=" * 60)
print("  CONFUSION MATRIX")
print("=" * 60)
print(f"  {'':>15}", "  ".join(f"{c:>8}" for c in le.classes_))
for i, row_label in enumerate(le.classes_):
    print(f"  Aktual {row_label:<8}", "  ".join(f"{v:>8}" for v in cm[i]))
print()

# ─────────────────────────────────────────────
# Tabel Hasil Prediksi Per Kabupaten
# ─────────────────────────────────────────────
df_hasil = df_raw.copy()
df_hasil["Prediksi_Risiko"] = y_pred
df_hasil["Benar"]           = df_hasil[TARGET] == df_hasil["Prediksi_Risiko"]

df_summary = df_hasil.groupby("Kabupaten_Kota").agg(
    Total_Bencana    = ("Jumlah_Bencana",  "sum"),
    Rerata_Bencana   = ("Jumlah_Bencana",  "mean"),
    Aktual_Dominan   = (TARGET,            lambda x: x.value_counts().index[0]),
    Prediksi_Risiko  = ("Prediksi_Risiko", lambda x: x.value_counts().index[0]),
    Akurasi_Lokal    = ("Benar",           "mean")
).reset_index()

df_summary["Rerata_Bencana"] = df_summary["Rerata_Bencana"].round(2)
df_summary["Akurasi_Lokal"]  = (df_summary["Akurasi_Lokal"] * 100).round(1)
df_summary = df_summary.sort_values("Total_Bencana", ascending=False).reset_index(drop=True)
df_summary.index += 1

print("=" * 70)
print("  HASIL PREDIKSI RISIKO BENCANA PER KABUPATEN/KOTA")
print(f"  Model: KNN (k={best_k})  |  Akurasi = {acc_full*100:.2f}%")
print("=" * 70)
print(df_summary.to_string())
print()

# ─────────────────────────────────────────────
# VISUALISASI
# ─────────────────────────────────────────────
WARNA = {"Rendah": "#2ca02c", "Sedang": "#ff7f0e", "Tinggi": "#d62728"}

fig, ax = plt.subplots(figsize=(12, 6))

warna_bar = [WARNA[k] for k in df_summary["Prediksi_Risiko"]]
bars = ax.barh(
    df_summary["Kabupaten_Kota"],
    df_summary["Total_Bencana"],
    color=warna_bar, edgecolor="white", linewidth=0.6
)

for bar, val, kat in zip(bars, df_summary["Total_Bencana"], df_summary["Prediksi_Risiko"]):
    ax.text(bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val} kejadian  [{kat}]",
            va="center", fontsize=9)

ax.invert_yaxis()
ax.set_xlabel("Total Kejadian Bencana (2010–2024)", fontsize=10)
ax.set_title(
    f"Prediksi Kategori Risiko Bencana per Kabupaten/Kota — Sulawesi Tengah\n"
    f"Model: KNN (k={best_k})  |  Akurasi CV = {acc_cv.mean()*100:.2f}%"
    f"  |  F1-Score = {f1_cv.mean():.4f}",
    fontsize=11, fontweight="bold"
)

legend_els = [mpatches.Patch(facecolor=v, label=k) for k, v in WARNA.items()]
ax.legend(handles=legend_els, title="Kategori Risiko", loc="lower right", fontsize=9)
ax.set_xlim(left=0, right=df_summary["Total_Bencana"].max() + 45)
ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()