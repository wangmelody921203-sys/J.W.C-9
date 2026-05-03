function doPost(e) {
  try {
    var payload = JSON.parse(e.postData.contents || "{}");
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("feedback") || SpreadsheetApp.getActiveSpreadsheet().insertSheet("feedback");

    ensureHeader_(sheet);

    var summary = payload.summary || {};
    sheet.appendRow([
      new Date(),
      payload.received_at || "",
      payload.source || "feedback.html",
      payload.ip_hint || "",
      payload.accuracy || "",
      payload.satisfaction || "",
      payload.comment || "",
      summary.emotion || "unknown",
      summary.share || "",
      summary.timestamp || "",
    ]);

    return jsonResponse_({ ok: true, stored: "sheet" });
  } catch (err) {
    return jsonResponse_({ ok: false, error: String(err) });
  }
}

function doGet() {
  return jsonResponse_({ ok: true, service: "feedback-webhook" });
}

function ensureHeader_(sheet) {
  if (sheet.getLastRow() > 0) {
    return;
  }

  sheet.appendRow([
    "server_received_at",
    "client_received_at",
    "source",
    "ip_hint",
    "accuracy",
    "satisfaction",
    "comment",
    "emotion",
    "share",
    "scan_timestamp",
  ]);

  sheet.getRange(1, 1, 1, 10).setFontWeight("bold");
  sheet.setFrozenRows(1);
}

function jsonResponse_(data) {
  return ContentService
    .createTextOutput(JSON.stringify(data))
    .setMimeType(ContentService.MimeType.JSON);
}
