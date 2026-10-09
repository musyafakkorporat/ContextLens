# ContextLens

ContextLens adalah aplikasi berbasis web untuk membantu menganalisis konteks dan kualitas informasi dari teks berbahasa Indonesia. Aplikasi ini menggabungkan klasifikasi teks, pengukuran ketidakpastian model, pencarian referensi, dan analisis bahasa menggunakan AI.

ContextLens merupakan alat bantu analisis, bukan penentu kebenaran otomatis. Hasil model, kemiripan referensi, dan interpretasi AI perlu diperiksa bersama bukti atau sumber yang relevan.

## Fitur Utama

- **Toxicity classification**: memperkirakan indikasi konten toxic menggunakan model IndoBERTweet.
- **Polarization classification**: memperkirakan indikasi konten polarized.
- **Probability calibration**: menampilkan probability model setelah kalibrasi.
- **Uncertainty analysis**: memberi penanda saat hasil model perlu diperhatikan karena ketidakpastian.
- **MAFINDO reference retrieval**: mencari referensi yang memiliki kemiripan semantik dengan teks masukan.
- **Context analysis**: menggunakan Gemini untuk membuat ringkasan, mengidentifikasi klaim utama, menguraikan pola penalaran, dan menyebutkan bukti yang diperlukan untuk memeriksa klaim.
- **Web interface**: menyediakan antarmuka untuk memasukkan teks dan membaca hasil analisis.

## Teknologi

| Komponen | Teknologi |
|---|---|
| Frontend | Next.js, React, TypeScript, Tailwind CSS |
| Backend API | FastAPI, Uvicorn |
| Klasifikasi teks | IndoBERTweet, Transformers, PyTorch |
| Semantic search | Sentence-Transformers |
| Analisis konteks | Google Gemini API |
| Referensi | Dataset MAFINDO |
| Penyimpanan model dan referensi | Hugging Face Hub |

## Arsitektur

```text
Pengguna
   |
   v
Next.js frontend
   |
   v
FastAPI: POST /analyze
   |
   +--> Toxicity classification
   |
   +--> Polarization classification
   |
   +--> Calibration dan uncertainty
   |
   +--> MAFINDO semantic search
   |
   +--> Gemini context analysis
   |
   v
Response JSON
   |
   v
Hasil ditampilkan pada frontend
```

## Repository Model dan Referensi

Model dan data referensi tidak disimpan langsung di repository GitHub karena ukurannya besar. Backend mengambil aset dari repository Hugging Face berikut:

- Toxicity model: `muss05/contextlens-toxicity`
- Polarization model: `muss05/contextlens-polarization`
- MAFINDO reference dataset: `muss05/contextlens-mafindo`

Ketiga repository tersebut bersifat **private**. Akun Hugging Face yang menjalankan backend harus memiliki akses baca ke repository tersebut.

## Persyaratan

Siapkan:

- Python yang sesuai dengan dependency pada `requirements.txt`.
- Node.js dan npm.
- Git.
- Google Gemini API key yang valid.
- Hugging Face read token yang memiliki akses ke repository private ContextLens.
- Koneksi internet untuk mengunduh aset Hugging Face dan menggunakan Gemini API.

## Instalasi Backend

Clone repository:

```powershell
git clone https://github.com/musyafakkorporat/ContextLens.git
cd ContextLens
```

Buat dan aktifkan virtual environment pada Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Pasang dependency:

```powershell
python -m pip install -r requirements.txt
```

Buat file `.env` berdasarkan template jika file tersebut belum tersedia:

```powershell
Copy-Item .env.example .env
```

Isi `.env` dengan kredensial milik sendiri:

```env
GEMINI_API_KEY=ISI_GEMINI_API_KEY
HF_TOKEN=ISI_HUGGING_FACE_READ_TOKEN
```

- `GEMINI_API_KEY` adalah API key dari Google AI Studio.
- `HF_TOKEN` adalah Hugging Face access token dengan izin baca ke repository private ContextLens.

Jangan menaruh nilai asli pada `.env.example`, jangan mengunggah `.env` ke GitHub, dan jangan membagikan token melalui chat atau tangkapan layar.

## Menjalankan Backend

Dari root project, dengan virtual environment aktif, jalankan:

```powershell
uvicorn --app-dir src api:app
```

Backend berjalan di `http://127.0.0.1:8000` secara default.

Dokumentasi interaktif API tersedia di `http://127.0.0.1:8000/docs`.

Endpoint yang tersedia:

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/` | Menampilkan pesan status API |
| `GET` | `/health` | Memeriksa kesehatan API |
| `POST` | `/analyze` | Menganalisis teks |

Contoh request untuk `POST /analyze`:

```json
{
  "text": "Ekonomi sedang sulit dan pemerintah perlu membantu masyarakat."
}
```

Response JSON berisi hasil toxicity, polarization, referensi MAFINDO jika tersedia, dan analisis konteks dari Gemini.

## Menjalankan Frontend

Buka terminal PowerShell kedua, lalu pindah ke folder frontend:

```powershell
cd frontend
npm install
```

Buat file `frontend/.env.local` jika belum tersedia, dengan isi:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Jalankan frontend:

```powershell
npm run dev
```

Buka `http://localhost:3000` di browser. Pastikan backend tetap berjalan di terminal pertama saat frontend digunakan.

Untuk memeriksa production build frontend, jalankan dari folder `frontend`:

```powershell
npm run build
```

## Catatan Interpretasi dan Privasi

- Probability model bukan probabilitas bahwa klaim pengguna benar atau salah.
- Status ketidakpastian bukan jaminan bahwa prediksi model benar atau salah.
- Semantic similarity hanya menunjukkan kemiripan teks dengan referensi yang ditemukan.
- Label pada data MAFINDO tidak otomatis menentukan kebenaran klaim baru.
- Analisis Gemini adalah bantuan interpretasi dan tetap perlu dibandingkan dengan bukti yang relevan.
- Teks yang dianalisis dikirim ke Google Gemini API untuk analisis konteks. Hindari memasukkan data pribadi atau informasi sensitif.

## Status Proyek

ContextLens memiliki frontend Next.js dan backend FastAPI yang terhubung. Endpoint analisis telah diuji secara lokal dan frontend telah berhasil melewati production build. Versi ini ditujukan untuk penggunaan lokal; deployment publik belum dikonfigurasi.

## Lisensi

Lisensi proyek ini belum ditentukan. Penggunaan model, dataset, serta layanan pihak ketiga tetap mengikuti lisensi dan ketentuan masing-masing sumber.
