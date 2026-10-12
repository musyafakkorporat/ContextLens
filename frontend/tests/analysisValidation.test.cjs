/* eslint-disable @typescript-eslint/no-require-imports -- This Node.js test file intentionally uses CommonJS. */
const test = require("node:test");
const assert = require("node:assert/strict");

const {
  validateAnalysisText,
  getApiErrorMessage,
} = require("../src/lib/analysisValidation.js");

test("menolak teks kosong", () => {
  assert.equal(
    validateAnalysisText(""),
    "Masukkan teks terlebih dahulu."
  );
});

test("menolak teks yang hanya berisi spasi", () => {
  assert.equal(
    validateAnalysisText("   "),
    "Masukkan teks terlebih dahulu."
  );
});

test("menerima tepat 5000 karakter", () => {
  assert.equal(validateAnalysisText("a".repeat(5000)), null);
});

test("menolak 5001 karakter", () => {
  assert.match(
    validateAnalysisText("a".repeat(5001)),
    /maksimal 5\.000 karakter/
  );
});

test("mempertahankan pesan error berbentuk string", () => {
  assert.equal(
    getApiErrorMessage({ detail: "Permintaan tidak valid." }),
    "Permintaan tidak valid."
  );
});

test("mengubah error panjang teks menjadi pesan yang jelas", () => {
  const data = {
    detail: [{
      type: "string_too_long",
      loc: ["body", "text"],
      msg: "String should have at most 5000 characters",
    }],
  };

  assert.match(getApiErrorMessage(data), /maksimal 5\.000 karakter/);
});

test("mengambil pesan validasi dari API", () => {
  const data = {
    detail: [{
      type: "value_error",
      loc: ["body", "text"],
      msg: "Value error, Teks tidak boleh kosong.",
    }],
  };

  assert.equal(
    getApiErrorMessage(data),
    "Teks tidak boleh kosong."
  );
});

test("menangani respons error dengan format yang tidak dikenal", () => {
  assert.equal(
    getApiErrorMessage({ unexpected: true }),
    "Terjadi kesalahan saat menganalisis teks."
  );
});
