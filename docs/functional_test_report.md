# ContextLens — Laporan Pengujian Fungsional

**Tanggal pengujian:** 12 Oktober 2026  
**Versi kode acuan:** `0d13fa9` — Filter repetitive text before MAfindo retrieval

## 1. Tujuan

Pengujian dilakukan untuk memeriksa fungsi utama ContextLens, meliputi validasi input, analisis toxicity dan polarization, analisis konteks oleh Gemini, serta pencarian referensi MAfindo.

Pengujian fungsional ini memeriksa perilaku aplikasi pada contoh yang diuji. Hasilnya tidak menjadi bukti bahwa semua prediksi model benar atau bahwa seluruh fungsi bebas dari bug.

## 2. Lingkungan Pengujian

- Sistem operasi: Windows
- Backend: Python, FastAPI, dan Uvicorn
- Frontend: Next.js 16.4.0
- Model klasifikasi: ContextLens leakage-free v2
- Model analisis konteks: Gemini
- Referensi: dataset MAfindo melalui semantic similarity

## 3. Hasil Pengujian Fungsional

| ID | Skenario | Hasil Aktual | Status |
|---|---|---|---|
| FT-01 | Analisis opini tentang evaluasi kebijakan | Toxicity dan polarization, Gemini, serta antarmuka menampilkan hasil | Lulus |
| FT-02 | Teks berisi penghinaan personal | Skor toxicity terkalibrasi 78,5%; label terindikasi toxic | Lulus fungsional |
| FT-03 | Teks yang menyebut pendukung kebijakan sebagai musuh rakyat | Toxicity 67,2%; polarization 80,0%; keduanya terindikasi | Lulus fungsional |
| FT-04 | Input kosong | Frontend menampilkan "Masukkan teks terlebih dahulu." | Lulus |
| FT-05 | Input 5.001 karakter melalui API | Backend menolak permintaan dengan HTTP 422 | Lulus |
| FT-06 | Input 5.001 karakter melalui frontend | Frontend menampilkan pesan bahwa teks maksimal 5.000 karakter | Lulus |
| FT-07 | Input tepat 5.000 karakter | Permintaan diterima dan hasil analisis ditampilkan | Lulus |
| FT-08 | Input repetitif berupa huruf `a` | Pencarian MAfindo mengembalikan referensi kosong | Lulus |
| FT-09 | Klaim terkait Anies Baswedan | Referensi terkait ditemukan dengan similarity sekitar 70,06% | Lulus untuk contoh ini |
| FT-10 | Teks mengenai harga beras dan pasokan | Dua referensi terkait ditemukan dengan similarity sekitar 71,29% dan 62,83% | Lulus untuk contoh ini |
| FT-11 | Pemeriksaan sintaks backend | `py_compile` selesai tanpa error | Lulus |
| FT-12 | Build frontend | `npm run build` selesai dengan pemeriksaan TypeScript | Lulus |
| FT-13 | Pemeriksaan perubahan Git | `git diff --check` bersih; commit perbaikan telah dibuat | Lulus |

## 4. Evaluasi Model pada Test Set Leakage-Free v2

Metrik ini berasal dari evaluasi test set terpisah. Metrik tersebut tidak boleh disamakan dengan hasil pengujian fungsional di atas.

| Metrik | Toxicity | Polarization |
|---|---:|---:|
| Jumlah data test | 3.654 | 3.522 |
| Accuracy | 91,60% | 84,72% |
| Precision | 46,24% | 45,61% |
| Recall | 56,94% | 47,46% |
| F1-score | 51,04% | 46,52% |

Kalibrasi temperatur memperbaiki Brier score, log loss, dan expected calibration error pada kedua model dibandingkan probabilitas mentah.

Untuk selective prediction polarization, coverage test sekitar 85,46%, sedangkan error pada prediksi yang diterima sekitar 11,16%. Nilai error test tersebut melampaui target error maksimum 10% yang digunakan ketika memilih konfigurasi pada validation set. Parameter tidak diubah menggunakan test set.

## 5. Keterbatasan

1. Pengujian fungsional dilakukan pada sejumlah contoh, bukan seluruh kemungkinan input.
2. Skor model merupakan sinyal klasifikasi, bukan bukti kebenaran atau kesalahan suatu klaim.
3. Similarity MAfindo tidak menjamin bahwa referensi relevan secara faktual. Label pada dataset juga bukan verifikasi independen atas klaim pengguna.
4. Penyaringan ketidakpastian toxicity saat ini tidak aktif karena `TOXICITY_UNCERTAIN_DELTA = 0.00`.
5. Model masih dapat menghasilkan skor untuk teks yang tidak bermakna. Hasil klasifikasi dan analisis Gemini pada input semacam itu perlu ditafsirkan dengan hati-hati.
6. Hasil pengujian ini tidak membuktikan kesiapan untuk deployment publik atau keamanan menyeluruh.

## 6. Kesimpulan

Fungsi utama ContextLens berhasil dijalankan pada skenario yang telah diuji. Validasi input, analisis model, penanganan batas panjang, analisis Gemini, dan retrieval MAfindo menunjukkan hasil sesuai dengan tujuan pengujian pada contoh tersebut.

Pengembangan selanjutnya perlu memprioritaskan evaluasi lebih luas, penanganan teks tidak bermakna pada seluruh pipeline, dan peninjauan kualitas relevansi referensi MAfindo.
