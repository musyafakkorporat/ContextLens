# ContextLens

ContextLens adalah prototipe analisis teks berbahasa Indonesia yang menggabungkan klasifikasi teks, kalibrasi probabilitas, analisis ketidakpastian, pencarian referensi MAFINDO, dan analisis konteks menggunakan Google Gemini.

Aplikasi memiliki **frontend web dan backend API yang dijalankan secara lokal**. Frontend dibuka melalui browser di `http://localhost:3000`, sedangkan backend berjalan di `http://127.0.0.1:8000`. Proyek ini tidak memerlukan deployment publik.

ContextLens merupakan alat bantu analisis, bukan sistem yang otomatis menentukan apakah suatu klaim benar atau salah. Hasil model, kemiripan referensi, dan interpretasi Gemini tetap perlu diperiksa dengan bukti serta sumber yang relevan.

## Fitur Utama

- **Toxicity classification**: memperkirakan indikasi konten toxic menggunakan model IndoBERTweet.
- **Polarization classification**: memperkirakan indikasi konten polarized.
- **Probability calibration**: menampilkan probabilitas model setelah kalibrasi.
- **Uncertainty analysis**: menandai hasil model yang perlu diperhatikan karena ketidakpastian.
- **MAFINDO reference retrieval**: mencari referensi yang memiliki kemiripan semantik dengan teks masukan.
- **Gemini context analysis**: membuat ringkasan, mengidentifikasi klaim utama, mengategorikan isi teks, menjelaskan kemungkinan pola penalaran, dan menyebutkan bukti atau konteks yang dibutuhkan.
- **Local web interface**: memasukkan teks melalui browser dan melihat hasil analisis dari backend.

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
        +--> Toxicity classification
        |
        +--> Polarization classification
        |
        +--> Calibration dan uncertainty analysis
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
├── src/
│   ├── api.py
│   ├── contextlens_pipeline.py
│   └── calibration_config.py
├── research/
│   └── scripts/
│       ├── analyze_selective_prediction.py
│       ├── analyze_uncertainty.py
│       ├── calibrate_models.py
│       ├── evaluate_calibration_test.py
│       ├── evaluate_pipeline.py
│       ├── train_toxicity_5000.py
│       └── train_polarized_5000.py
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx
│   │       ├── layout.tsx
│   │       └── globals.css
│   ├── package.json
│   ├── .env.example
│   └── .env.local          # lokal, tidak di-commit
├── requirements.txt
├── .env.example
└── README.md
```

Folder `src/` berisi kode utama yang dibutuhkan untuk menjalankan backend. Folder `research/scripts/` menyimpan script penelitian, persiapan data, pelatihan, eksperimen, dan evaluasi; script di folder tersebut tidak perlu dijalankan saat menggunakan aplikasi utama.

Model dan data referensi berukuran besar tidak disimpan langsung di GitHub. Aset tersebut berada di Hugging Face Hub.

## Repository Model dan Referensi

Backend menggunakan tiga repository Hugging Face berikut:

- Toxicity model: [`muss05/contextlens-toxicity`](https://huggingface.co/muss05/contextlens-toxicity)
- Polarization model: [`muss05/contextlens-polarization`](https://huggingface.co/muss05/contextlens-polarization)
- Referensi MAFINDO: [`muss05/contextlens-mafindo`](https://huggingface.co/datasets/muss05/contextlens-mafindo)

Repository tersebut bersifat **private**. Pengguna perlu mempunyai izin baca untuk mengaksesnya. Backend juga memerlukan koneksi internet untuk mengunduh aset yang belum ada di cache dan untuk memanggil Gemini API.

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

`GEMINI_API_KEY` digunakan untuk mengakses Gemini API. `HF_TOKEN` harus memiliki izin baca ke repository private model dan referensi ContextLens. Untuk penggunaan lokal, token Hugging Face yang sudah disimpan melalui `hf auth login` juga dapat digunakan oleh Hugging Face Hub jika `HF_TOKEN` tidak disetel di environment.

**Jangan commit atau mengunggah `.env` ke GitHub.** File `.env.example` hanya template kosong dan aman untuk dibagikan.

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

Response mencakup hasil toxicity, polarization, referensi MAFINDO jika tersedia, dan analisis konteks dari Gemini.

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

## Catatan Interpretasi Hasil

- Probability model menunjukkan keluaran model, bukan probabilitas bahwa suatu klaim benar atau salah.
- Status uncertainty bukan jaminan bahwa prediksi model benar.
- Semantic similarity menunjukkan kemiripan dengan referensi, bukan bukti bahwa teks pengguna memiliki fakta atau konteks yang sama.
- Label pada data referensi MAFINDO tidak otomatis menentukan kebenaran klaim baru.
- Analisis Gemini merupakan bantuan interpretasi dan tetap memerlukan pemeriksaan sumber serta bukti yang relevan.

## Status Proyek

ContextLens dikembangkan sebagai proyek yang dijalankan secara lokal. Backend FastAPI dan endpoint `/analyze` telah diuji, serta frontend Next.js telah berhasil melewati production build. Deployment publik tidak dikonfigurasi dalam versi ini.

## Lisensi

Lisensi proyek belum ditentukan. Penggunaan model, dataset, dan layanan pihak ketiga tetap mengikuti lisensi serta ketentuan masing-masing sumber.
