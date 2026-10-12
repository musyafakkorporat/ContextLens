# ContextLens

ContextLens adalah prototipe analisis teks berbahasa Indonesia yang menggabungkan klasifikasi toxicity, klasifikasi polarization, kalibrasi probabilitas, analisis ketidakpastian, pencarian referensi MAFINDO, dan analisis konteks menggunakan Google Gemini.

Aplikasi memiliki **frontend web dan backend API yang dijalankan secara lokal**. Frontend dibuka melalui browser di `http://localhost:3000`, sedangkan backend berjalan di `http://127.0.0.1:8000`. Proyek ini tidak memerlukan deployment publik.

ContextLens merupakan alat bantu analisis, bukan sistem yang otomatis menentukan apakah suatu klaim benar atau salah. Hasil model, kemiripan referensi, dan interpretasi Gemini tetap perlu diperiksa dengan bukti serta sumber yang relevan.

## Fitur Utama

- **Toxicity classification:** memperkirakan indikasi konten toxic menggunakan model IndoBERTweet.
- **Polarization classification:** memperkirakan indikasi konten polarized.
- **Probability calibration:** menampilkan probabilitas model setelah kalibrasi temperatur.
- **Uncertainty analysis:** status ketidakpastian polarization menggunakan delta `0.26`. Penyaringan ketidakpastian toxicity saat ini nonaktif karena delta toxicity disetel ke `0.00`.
- **Input validation:** menolak input kosong dan input yang melebihi 5.000 karakter.
- **MAfindo reference retrieval:** mencari referensi berdasarkan semantic similarity. Teks yang sangat repetitif dilewati agar tidak menghasilkan referensi yang menyesatkan.
- **Gemini context analysis:** membuat ringkasan, mengidentifikasi klaim utama, mengategorikan isi teks, menjelaskan kemungkinan pola penalaran, serta menyebutkan bukti atau konteks yang dibutuhkan.
- **Local web interface:** memasukkan teks melalui browser dan melihat hasil analisis dari backend.

## Teknologi

| Komponen | Teknologi |
|---|---|
| Frontend | Next.js, React, TypeScript, Tailwind CSS |
| Backend API | FastAPI, Uvicorn |
| Klasifikasi teks | IndoBERTweet, Transformers, PyTorch |
| Semantic search | Sentence-Transformers, NumPy |
| Pengolahan data | Pandas |
| Analisis konteks | Google Gemini API |
| Model dan data referensi | Hugging Face Hub |

## Alur Analisis

```text
Pengguna memasukkan teks
        |
        v
Frontend Next.js (localhost:3000)
        |
        v
FastAPI /analyze (localhost:8000)
        |
        +--> Validasi input
        |
        +--> Toxicity classification
        |
        +--> Polarization classification
        |
        +--> Kalibrasi probabilitas
        |
        +--> Analisis ketidakpastian
        |
        +--> Pencarian referensi MAFINDO
        |
        +--> Analisis konteks dengan Gemini API
        |
        v
Hasil analisis dalam JSON
        |
        v
Hasil ditampilkan pada frontend
```

## Struktur Proyek

```text
ContextLens/
|-- src/
|   |-- api.py
|   |-- contextlens_pipeline.py
|   |-- text_quality.py
|   `-- calibration_config.py
|-- research/
|   `-- scripts/
|       |-- rebuild_leakage_free_splits.py
|       |-- train_toxicity_leakage_free_v2.py
|       |-- train_polarized_leakage_free_v2.py
|       |-- calibrate_leakage_free_v2.py
|       |-- analyze_selective_prediction_leakage_free_v2.py
|       |-- evaluate_leakage_free_v2_validation.py
|       `-- evaluate_leakage_free_v2_test.py
|-- tests/
|   `-- test_text_quality.py
|-- frontend/
|   |-- src/app/
|   |   |-- page.tsx
|   |   |-- layout.tsx
|   |   `-- globals.css
|   `-- package.json
|-- docs/
|   `-- functional_test_report.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

Folder `src/` berisi kode utama backend. Folder `research/scripts/` menyimpan skrip persiapan data, pelatihan, kalibrasi, dan evaluasi. Folder `tests/` berisi tes otomatis untuk pemeriksaan teks repetitif. Folder `docs/` berisi laporan pengujian fungsional.

Dataset hasil pemrosesan, model lokal, dan hasil eksperimen tidak perlu diunggah ke GitHub. File `.env` dan `frontend/.env.local` digunakan secara lokal dan tidak boleh berisi kredensial yang ikut di-commit.

## Repository Model dan Referensi

Backend menggunakan tiga repository Hugging Face berikut:

- **Toxicity model v2:** [`muss05/contextlens-toxicity-leakage-free-v2`](https://huggingface.co/muss05/contextlens-toxicity-leakage-free-v2)
- **Polarization model v2:** [`muss05/contextlens-polarization-leakage-free-v2`](https://huggingface.co/muss05/contextlens-polarization-leakage-free-v2)
- **Referensi MAFINDO:** [`muss05/contextlens-mafindo`](https://huggingface.co/datasets/muss05/contextlens-mafindo)

Repository model dan dataset tersebut bersifat **private**. Pengguna perlu memiliki izin baca untuk mengaksesnya. Backend memerlukan koneksi internet untuk mengunduh aset yang belum tersedia di cache dan untuk memanggil Gemini API.

Model lama tetap tersimpan di repository terpisah dan tidak digunakan oleh konfigurasi runtime v2 saat ini.

## Persyaratan

Siapkan perangkat dengan:

- Python dan pip.
- Node.js dan npm.
- Git.
- Google Gemini API key yang valid.
- Akun Hugging Face dengan akses baca ke repository private ContextLens.
- Koneksi internet.

## Menjalankan Backend secara Lokal

### 1. Clone repository

```powershell
git clone https://github.com/musyafakkorporat/ContextLens.git
cd ContextLens
```

### 2. Buat virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Pasang dependency

```powershell
python -m pip install -r requirements.txt
```

### 4. Siapkan kredensial

Buat file `.env` di root proyek berdasarkan `.env.example`, lalu isi dengan kredensial sendiri:

```env
GEMINI_API_KEY=ISI_GEMINI_API_KEY
HF_TOKEN=ISI_HUGGING_FACE_READ_TOKEN
```

`GEMINI_API_KEY` digunakan untuk mengakses Gemini API. `HF_TOKEN` harus memiliki izin baca ke repository private model dan referensi ContextLens.

Token Hugging Face yang sudah disimpan melalui `hf auth login` juga dapat digunakan oleh Hugging Face Hub jika `HF_TOKEN` tidak disetel di environment.

**Jangan commit atau mengunggah `.env` ke GitHub.** File `.env.example` hanya template dan tidak boleh berisi token asli.

### 5. Jalankan backend

Dari root proyek, jalankan:

```powershell
uvicorn --app-dir src api:app
```

Backend berjalan pada `http://127.0.0.1:8000` secara default. Dokumentasi interaktif API tersedia di `http://127.0.0.1:8000/docs`.

Endpoint yang digunakan:

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/health` | Memeriksa status backend |
| `POST` | `/analyze` | Menganalisis teks |

Contoh body request untuk `/analyze`:

```json
{
  "text": "Ekonomi sedang sulit dan pemerintah perlu membantu masyarakat."
}
```

Response mencakup hasil toxicity, polarization, referensi MAFINDO jika tersedia, serta analisis konteks dari Gemini.

## Menjalankan Frontend secara Lokal

Buka **PowerShell kedua** dan pindah ke folder frontend:

```powershell
cd frontend
npm install
```

Buat file `frontend/.env.local` jika belum tersedia, dengan isi:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Kemudian jalankan:

```powershell
npm run dev
```

Buka `http://localhost:3000` di browser. Pastikan backend tetap berjalan di terminal pertama selama frontend digunakan.

Untuk memeriksa build produksi frontend, jalankan dari folder `frontend`:

```powershell
npm run build
```

## Menjalankan Tes Otomatis

Aktifkan virtual environment, lalu jalankan perintah berikut dari root proyek:

```powershell
python -m unittest discover -s tests -v
```

Tes menggunakan `unittest` bawaan Python. Tes saat ini mencakup teks repetitif, kata berulang, teks normal, input kosong, whitespace, dan tanda baca. Tes unit ini tidak perlu memuat model besar atau memanggil Gemini API.

## Pengujian Fungsional

Laporan pengujian fungsional tersedia di [`docs/functional_test_report.md`](docs/functional_test_report.md).

Skenario yang telah diuji mencakup analisis teks biasa, input kosong, batas maksimum 5.000 karakter, penolakan input 5.001 karakter, serta pencarian referensi MAFINDO untuk teks repetitif dan teks bermakna. Pengujian ini merupakan pemeriksaan fungsional pada sejumlah contoh, bukan jaminan bahwa seluruh keluaran model benar.

## Catatan Interpretasi Hasil

- Probability model adalah skor prediksi model, bukan probabilitas bahwa sebuah klaim benar atau salah.
- Status ketidakpastian bukan jaminan bahwa prediksi model benar.
- Penyaringan ketidakpastian toxicity saat ini nonaktif karena `TOXICITY_UNCERTAIN_DELTA = 0.00`.
- Semantic similarity menunjukkan kemiripan teks dengan referensi, bukan bukti bahwa teks pengguna memiliki fakta atau konteks yang sama.
- Filter teks repetitif melewati pencarian MAFINDO untuk input yang sangat repetitif, tetapi tidak menjamin bahwa seluruh referensi lain pasti relevan.
- Label pada data referensi MAFINDO tidak otomatis menentukan kebenaran klaim baru.
- Analisis Gemini merupakan bantuan interpretasi dan tetap memerlukan pemeriksaan sumber serta bukti yang relevan.

## Keterbatasan Model

Model toxicity dan polarization masih memiliki keterbatasan. Pada evaluasi test set leakage-free v2:

| Metrik | Toxicity | Polarization |
|---|---:|---:|
| Jumlah data test | 3.654 | 3.522 |
| Accuracy | 91,60% | 84,72% |
| Precision | 46,24% | 45,61% |
| Recall | 56,94% | 47,46% |
| F1-score | 51,04% | 46,52% |

Untuk selective prediction polarization, coverage test sekitar 85,46%, dengan error pada prediksi yang diterima sekitar 11,16%. Nilai tersebut melampaui target error maksimum 10% yang digunakan saat memilih konfigurasi pada validation set. Test set tidak digunakan untuk mengubah parameter konfigurasi.

Metrik tersebut menunjukkan bahwa hasil model perlu ditafsirkan secara hati-hati, khususnya untuk penggunaan yang sensitif.

## Status Proyek

ContextLens dikembangkan sebagai prototipe yang dijalankan secara lokal. Backend FastAPI dan endpoint `/analyze` telah diuji, frontend Next.js berhasil melewati production build, dan sejumlah skenario pengujian fungsional telah dijalankan.

Deployment publik belum dikonfigurasi dalam versi ini.

## Lisensi

Lisensi proyek belum ditentukan. Penggunaan model, dataset, dan layanan pihak ketiga tetap mengikuti lisensi serta ketentuan masing-masing sumber.
