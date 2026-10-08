"use client";

import { FormEvent, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

type Reference = {
  title: string;
  content: string;
  is_hoax: number;
  similarity: number;
};

type AnalysisResult = {
  text: string;

  toxicity: {
    label: number;
    probability: number;
    calibrated_probability: number;
    status: "confident" | "uncertain";
  };

  polarization: {
    label: number;
    probability: number;
    calibrated_probability: number;
    status: "confident" | "uncertain";
  };

  mafindo_references: Reference[];

  gemini: {
    summary: string;
    claim: string;
    content_type: "fact" | "opinion" | "prediction" | "mixed";
    reasoning_pattern: string;
    evidence_needed: string;
  };
};


function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}


function statusText(
  status: "confident" | "uncertain"
) {
  return status === "confident"
    ? "Cukup yakin"
    : "Perlu diperhatikan";
}


export default function Home() {

  const [text, setText] = useState("");

  const [result, setResult] = useState<AnalysisResult | null>(
    null
  );

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");


  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {

    event.preventDefault();

    setError("");
    setResult(null);

    if (!text.trim()) {
      setError("Masukkan teks terlebih dahulu.");
      return;
    }

    setLoading(true);

    try {

      const response = await fetch(
        `${API_URL}/analyze`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            text: text.trim()
          })
        }
      );

      const data = await response.json();

      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Terjadi kesalahan saat menganalisis teks."
        );
      }

      setResult(data);

    } catch (err) {

      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Tidak dapat terhubung ke ContextLens API."
        );
      }

    } finally {

      setLoading(false);
    }
  }


  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">

      <div className="mx-auto max-w-6xl px-5 py-8 sm:px-8 lg:px-10">

        {/* HEADER */}

        <header className="mb-8">

          <div className="flex items-center gap-3">

            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-slate-900 text-lg font-bold text-white">
              C
            </div>

            <div>

              <h1 className="text-2xl font-bold tracking-tight">
                ContextLens
              </h1>

              <p className="text-sm text-slate-500">
                Analisis konteks informasi dengan AI
              </p>

            </div>

          </div>

        </header>


        {/* INPUT */}

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">

          <div className="mb-5">

            <h2 className="text-xl font-semibold">
              Analisis sebuah teks
            </h2>

            <p className="mt-1 text-sm leading-6 text-slate-500">
              Masukkan komentar, postingan, atau informasi
              yang ingin dianalisis.
            </p>

          </div>


          <form onSubmit={handleSubmit}>

            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Contoh: Menurut saya harga BBM sekarang terlalu mahal dan pemerintah harus menurunkannya..."
              className="min-h-40 w-full resize-y rounded-xl border border-slate-300 bg-white p-4 text-sm leading-6 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
            />


            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">

              <p className="text-xs text-slate-400">
                ContextLens tidak otomatis menentukan bahwa
                sebuah klaim benar atau salah.
              </p>

              <button
                type="submit"
                disabled={loading}
                className="rounded-xl bg-slate-900 px-6 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
              >

                {loading
                  ? "Menganalisis..."
                  : "Analisis teks"}

              </button>

            </div>

          </form>


          {error && (

            <div className="mt-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>

          )}

        </section>


        {/* LOADING */}

        {loading && (

          <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

            <div className="flex items-center gap-3">

              <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-blue-600" />

              <div>

                <p className="font-medium">
                  Sedang menganalisis teks...
                </p>

                <p className="text-sm text-slate-500">
                  Model NLP, semantic search, dan Gemini
                  sedang memproses input.
                </p>

              </div>

            </div>

          </section>

        )}


        {/* RESULT */}

        {result && !loading && (

          <section className="mt-6 space-y-6">


            {/* ML */}

            <div>

              <div className="mb-4">

                <h2 className="text-xl font-semibold">
                  Hasil analisis
                </h2>

                <p className="text-sm text-slate-500">
                  Hasil model ditampilkan sebagai sinyal,
                  bukan keputusan mutlak.
                </p>

              </div>


              <div className="grid gap-4 md:grid-cols-2">


                {/* TOXICITY */}

                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">

                  <div className="flex items-start justify-between gap-4">

                    <div>

                      <p className="text-sm font-medium text-slate-500">
                        Toxicity
                      </p>

                      <p className="mt-2 text-3xl font-bold">
                        {formatPercent(
                          result.toxicity.calibrated_probability
                        )}
                      </p>

                    </div>


                    <span
                      className={`rounded-full px-3 py-1 text-xs font-semibold ${
                        result.toxicity.status === "confident"
                          ? "bg-emerald-50 text-emerald-700"
                          : "bg-amber-50 text-amber-700"
                      }`}
                    >
                      {statusText(
                        result.toxicity.status
                      )}
                    </span>

                  </div>


                  <div className="mt-5 space-y-2 text-sm">

                    <div className="flex justify-between">

                      <span className="text-slate-500">
                        Label
                      </span>

                      <span className="font-medium">
                        {result.toxicity.label}
                      </span>

                    </div>


                    <div className="flex justify-between">

                      <span className="text-slate-500">
                        Probability mentah
                      </span>

                      <span className="font-medium">
                        {formatPercent(
                          result.toxicity.probability
                        )}
                      </span>

                    </div>


                    <div className="flex justify-between">

                      <span className="text-slate-500">
                        Probability terkalibrasi
                      </span>

                      <span className="font-medium">
                        {formatPercent(
                          result.toxicity.calibrated_probability
                        )}
                      </span>

                    </div>

                  </div>

                </div>


                {/* POLARIZATION */}

                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">

                  <div className="flex items-start justify-between gap-4">

                    <div>

                      <p className="text-sm font-medium text-slate-500">
                        Polarization
                      </p>

                      <p className="mt-2 text-3xl font-bold">
                        {formatPercent(
                          result.polarization.calibrated_probability
                        )}
                      </p>

                    </div>


                    <span
                      className={`rounded-full px-3 py-1 text-xs font-semibold ${
                        result.polarization.status === "confident"
                          ? "bg-emerald-50 text-emerald-700"
                          : "bg-amber-50 text-amber-700"
                      }`}
                    >
                      {statusText(
                        result.polarization.status
                      )}
                    </span>

                  </div>


                  <div className="mt-5 space-y-2 text-sm">

                    <div className="flex justify-between">

                      <span className="text-slate-500">
                        Label
                      </span>

                      <span className="font-medium">
                        {result.polarization.label}
                      </span>

                    </div>


                    <div className="flex justify-between">

                      <span className="text-slate-500">
                        Probability mentah
                      </span>

                      <span className="font-medium">
                        {formatPercent(
                          result.polarization.probability
                        )}
                      </span>

                    </div>


                    <div className="flex justify-between">

                      <span className="text-slate-500">
                        Probability terkalibrasi
                      </span>

                      <span className="font-medium">
                        {formatPercent(
                          result.polarization.calibrated_probability
                        )}
                      </span>

                    </div>

                  </div>

                </div>

              </div>

            </div>


            {/* GEMINI */}

            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">

              <div className="mb-5">

                <h2 className="text-lg font-semibold">
                  Context Analysis
                </h2>

                <p className="text-sm text-slate-500">
                  Analisis bahasa dan konteks dari Gemini.
                </p>

              </div>


              <div className="grid gap-5 lg:grid-cols-2">


                <div>

                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Ringkasan
                  </p>

                  <p className="mt-2 text-sm leading-7 text-slate-700">
                    {result.gemini.summary}
                  </p>

                </div>


                <div>

                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Jenis isi
                  </p>

                  <span className="mt-2 inline-flex rounded-full bg-blue-50 px-3 py-1 text-sm font-semibold capitalize text-blue-700">
                    {result.gemini.content_type}
                  </span>

                </div>


                <div>

                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Klaim utama
                  </p>

                  <p className="mt-2 text-sm leading-7 text-slate-700">
                    {result.gemini.claim}
                  </p>

                </div>


                <div>

                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Pola penalaran
                  </p>

                  <p className="mt-2 text-sm leading-7 text-slate-700">
                    {result.gemini.reasoning_pattern}
                  </p>

                </div>


                <div className="lg:col-span-2">

                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Bukti / konteks yang dibutuhkan
                  </p>

                  <p className="mt-2 text-sm leading-7 text-slate-700">
                    {result.gemini.evidence_needed}
                  </p>

                </div>

              </div>

            </div>


            {/* MAFINDO */}

            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">

              <div className="mb-5">

                <h2 className="text-lg font-semibold">
                  Reference MAfindo
                </h2>

                <p className="mt-1 text-sm leading-6 text-slate-500">
                  Reference ditemukan berdasarkan semantic
                  similarity. Kemiripan bukan bukti bahwa
                  klaim pengguna benar atau salah.
                </p>

              </div>


              {result.mafindo_references.length === 0 ? (

                <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-5 text-sm text-slate-500">
                  Tidak ditemukan reference yang cukup relevan.
                </div>

              ) : (

                <div className="space-y-4">

                  {result.mafindo_references.map(
                    (reference, index) => (

                      <article
                        key={`${reference.title}-${index}`}
                        className="rounded-xl border border-slate-200 p-4"
                      >

                        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">

                          <h3 className="font-semibold text-slate-900">
                            {reference.title}
                          </h3>

                          <span className="shrink-0 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                            Similarity{" "}
                            {formatPercent(
                              reference.similarity
                            )}
                          </span>

                        </div>


                        <p className="mt-3 text-sm leading-6 text-slate-600">
                          {reference.content}
                        </p>


                        <p className="mt-3 text-xs text-slate-400">
                          Kategori dataset:{" "}
                          {reference.is_hoax}
                        </p>

                      </article>

                    )
                  )}

                </div>

              )}

            </div>


            {/* DISCLAIMER */}

            <div className="rounded-2xl border border-blue-100 bg-blue-50 p-5 text-sm leading-6 text-blue-900">

              <p className="font-semibold">
                Catatan ContextLens
              </p>

              <p className="mt-1">
                Hasil model, similarity reference, dan analisis
                AI tidak otomatis membuktikan kebenaran atau
                kesalahan sebuah klaim. Gunakan sumber resmi
                dan bukti yang relevan untuk verifikasi akhir.
              </p>

            </div>

          </section>

        )}

      </div>

    </main>
  );
}