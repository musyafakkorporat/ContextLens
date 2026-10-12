const EMPTY_TEXT_ERROR = "Masukkan teks terlebih dahulu.";
const TOO_LONG_TEXT_ERROR =
  "Teks maksimal 5.000 karakter. Kurangi panjang teks lalu coba lagi.";
const DEFAULT_API_ERROR =
  "Terjadi kesalahan saat menganalisis teks.";

function validateAnalysisText(text) {
  if (!text.trim()) {
    return EMPTY_TEXT_ERROR;
  }

  if (Array.from(text).length > 5000) {
    return TOO_LONG_TEXT_ERROR;
  }

  return null;
}

function getApiErrorMessage(data) {
  if (data && typeof data.detail === "string") {
    return data.detail;
  }

  if (data && Array.isArray(data.detail)) {
    const details = data.detail;

    const tooLong = details.some(
      (item) =>
        item &&
        item.type === "string_too_long" &&
        Array.isArray(item.loc) &&
        item.loc.includes("text")
    );

    if (tooLong) {
      return TOO_LONG_TEXT_ERROR;
    }

    const firstMessage = details.find(
      (item) => item && typeof item.msg === "string"
    )?.msg;

    if (firstMessage) {
      return firstMessage.replace(/^Value error,\s*/i, "");
    }
  }

  return DEFAULT_API_ERROR;
}

module.exports = {
  validateAnalysisText,
  getApiErrorMessage,
};
