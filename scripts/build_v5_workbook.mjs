import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "/workspace/scratch/82fa5d97afa7/retail_credit_pd_model";
const source = `${root}/outputs/v41/model_validation_pack_v41.xlsx`;
const outputDir = `${root}/outputs/v5`;
const outputPath = `${outputDir}/model_validation_pack_v5.xlsx`;
const previewDir = `${outputDir}/previews`;
const payload = JSON.parse(await fs.readFile(`${root}/reports/validation_payload.json`, "utf8"));
await fs.mkdir(previewDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const sourcePreview = await workbook.render({ sheetName: "Validation Summary", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(`${previewDir}/source_executive_summary.png`, new Uint8Array(await sourcePreview.arrayBuffer()));

const navy = "#123455", blue = "#1F4E78", pale = "#D9EAF7", gold = "#F2B705";
const green = "#E2F0D9", red = "#FCE4D6", gray = "#E7E6E6", font = "Arial";
const moneyFmt = '"VND "#,##0;[Red]("VND "#,##0);-';
const pctFmt = "0.0%";

function baseSheet(name, title, subtitle) {
  const s = workbook.worksheets.add(name);
  s.showGridLines = false;
  s.getRange("A2").values = [[title]];
  s.getRange("A2:H2").format = { font: { name: font, size: 15, bold: true, color: navy }, rowHeight: 25 };
  s.getRange("A3").values = [[subtitle]];
  s.getRange("A3:H3").format = { font: { name: font, size: 10, italic: true, color: "#666666" }, rowHeight: 20 };
  s.getRange("A4:H4").format.borders = { bottom: { style: "thin", color: gold } };
  return s;
}

function styleHeader(range) {
  range.format = { fill: navy, font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, rowHeight: 30 };
}

function styleBody(range) {
  range.format.font = { name: font, size: 10, color: "#1F1F1F" };
  range.format.verticalAlignment = "center";
  range.format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" } };
}

const summary = baseSheet("V5 Summary", "V5 Decision & Profitability Summary", "Walk-forward validation, portfolio economics, stress testing and policy simulation");
summary.getRange("A6:B16").values = [
  ["Metric", "Result"],
  ["Build", payload.v5_summary.build],
  ["Validation status", payload.v5_summary.status],
  ["Champion mean AUC", payload.v5_summary.champion_mean_auc],
  ["Challenger mean AUC", payload.v5_summary.challenger_mean_auc],
  ["Challenger decision", payload.v5_summary.challenger_decision],
  ["Recommended approve PD max", payload.v5_summary.recommended_approve_pd_max],
  ["Recommended review PD max", payload.v5_summary.recommended_review_pd_max],
  ["Approval rate", payload.v5_summary.recommended_approval_rate],
  ["Observed bad rate", payload.v5_summary.recommended_bad_rate],
  ["Expected profit", payload.v5_summary.recommended_expected_profit],
];
styleHeader(summary.getRange("A6:B6")); styleBody(summary.getRange("A7:B16"));
summary.getRange("B9:B10").format.numberFormat = "0.000";
summary.getRange("B12:B15").format.numberFormat = pctFmt;
summary.getRange("B16").format.numberFormat = moneyFmt;
summary.getRange("D6").values = [["Management conclusion"]]; styleHeader(summary.getRange("D6"));
summary.getRange("D7").values = [["The WOE logistic model remains the champion because the challenger does not deliver the required 0.02 mean-AUC uplift. The simulated candidate strategy increases controlled approvals while keeping observed bad rate below 10%. Severe-stress profitability remains positive in the synthetic portfolio."]];
summary.getRange("D7").format = { fill: pale, font: { name: font, size: 11, color: navy }, wrapText: true, verticalAlignment: "top", rowHeight: 82 };
summary.getRange("D13").values = [["Required before real deployment"]]; styleHeader(summary.getRange("D13"));
summary.getRange("D14").values = [["Validate on bank data; approve assumptions and cut-offs through Credit Committee; connect managed identity and database services; complete pilot, concurrency, security and penetration tests."]];
summary.getRange("D14").format = { fill: red, font: { name: font, size: 10, color: "#9C0006" }, wrapText: true, verticalAlignment: "top", rowHeight: 72 };
summary.getRange("A:H").format.columnWidth = 16; summary.getRange("A:A").format.columnWidth = 29; summary.getRange("D:D").format.columnWidth = 66;

const wf = baseSheet("Walk Forward", "Chronological Walk-Forward Validation", "Four expanding training windows; champion and challenger tested on the next unseen period");
const wfRows = payload.v5_champion_challenger;
wf.getRange("A6:J6").values = [["Fold", "Model", "Train end", "Test start", "Test end", "Applications", "AUC", "Gini", "KS", "Brier"]];
wf.getRangeByIndexes(6, 0, wfRows.length, 10).values = wfRows.map(r => [r.window, r.model, r.train_end.slice(0,10), r.test_start.slice(0,10), r.test_end.slice(0,10), r.applications, r.auc, r.gini, r.ks, r.brier]);
styleHeader(wf.getRange("A6:J6")); styleBody(wf.getRangeByIndexes(6,0,wfRows.length,10));
wf.getRange(`G7:J${6+wfRows.length}`).format.numberFormat = "0.000";
wf.getRange("A:A").format.columnWidth = 11; wf.getRange("B:B").format.columnWidth = 29; wf.getRange("C:E").format.columnWidth = 14; wf.getRange("F:J").format.columnWidth = 12;
wf.freezePanes.freezeRows(6);

const strat = baseSheet("Strategy Simulator", "Decision Strategy Simulator", "Edit the blue cells to test approval and review cut-offs");
strat.getRange("A5:B6").values = [["Approve PD max", 0.10], ["Review PD max", 0.18]];
strat.getRange("A5:A6").format = { fill: gray, font: { name: font, size: 10, bold: true, color: navy } };
strat.getRange("B5:B6").format = { fill: "#DDEBF7", font: { name: font, size: 11, bold: true, color: "#0000FF" }, numberFormat: pctFmt, horizontalAlignment: "center" };
strat.getRange("B5").dataValidation = { rule: { type: "list", values: ["6%","7%","8%","9%","10%","11%","12%","13%"] } };
strat.getRange("B6").dataValidation = { rule: { type: "list", values: ["14%","15%","16%","17%","18%","19%","20%","21%","22%"] } };
strat.getRange("D5:H5").values = [["Approval rate", "Review rate", "Observed bad rate", "Expected profit", "RAROC"]]; styleHeader(strat.getRange("D5:H5"));
const strategy = payload.v5_strategy_grid;
const end = 10 + strategy.length;
strat.getRange("A10:J10").values = [["Approve PD max", "Review PD max", "Approval rate", "Review rate", "Observed bad rate", "Mean PD", "Total EAD", "Expected loss", "Expected profit", "RAROC"]];
strat.getRangeByIndexes(10,0,strategy.length,10).values = strategy.map(r => [r.approve_pd_max,r.review_pd_max,r.approval_rate,r.review_rate,r.observed_bad_rate,r.mean_pd,r.total_ead,r.expected_loss,r.expected_profit,r.portfolio_raroc]);
styleHeader(strat.getRange("A10:J10")); styleBody(strat.getRangeByIndexes(10,0,strategy.length,10));
strat.getRange(`A11:F${end}`).format.numberFormat = pctFmt; strat.getRange(`G11:I${end}`).format.numberFormat = moneyFmt; strat.getRange(`J11:J${end}`).format.numberFormat = pctFmt;
strat.getRange("D6:H6").formulas = [[
  `=SUMIFS($C$11:$C$${end},$A$11:$A$${end},$B$5,$B$11:$B$${end},$B$6)`,
  `=SUMIFS($D$11:$D$${end},$A$11:$A$${end},$B$5,$B$11:$B$${end},$B$6)`,
  `=SUMIFS($E$11:$E$${end},$A$11:$A$${end},$B$5,$B$11:$B$${end},$B$6)`,
  `=SUMIFS($I$11:$I$${end},$A$11:$A$${end},$B$5,$B$11:$B$${end},$B$6)`,
  `=SUMIFS($J$11:$J$${end},$A$11:$A$${end},$B$5,$B$11:$B$${end},$B$6)`
]];
strat.getRange("D6:F6").format.numberFormat = pctFmt; strat.getRange("G6").format.numberFormat = moneyFmt; strat.getRange("H6").format.numberFormat = pctFmt;
strat.getRange("D6:H6").format = { fill: green, font: { name: font, size: 11, bold: true, color: navy }, horizontalAlignment: "center" };
strat.getRange("A:B").format.columnWidth = 17; strat.getRange("C:F").format.columnWidth = 16; strat.getRange("G:I").format.columnWidth = 19; strat.getRange("J:J").format.columnWidth = 14;
strat.freezePanes.freezeRows(10);

const stress = baseSheet("Stress Test", "Portfolio Stress Testing", "Base, Downturn and Severe scenarios apply shocks to PD odds, LGD and funding cost");
const sr = payload.v5_stress_testing;
stress.getRange("A6:I6").values = [["Scenario","PD odds multiplier","LGD","Funding rate","Approval rate","Portfolio PD","Expected loss","Expected profit","RAROC"]];
stress.getRangeByIndexes(6,0,sr.length,9).values = sr.map(r => [r.scenario,r.pd_odds_multiplier,r.lgd,r.funding_rate,r.approval_rate,r.portfolio_pd,r.expected_loss,r.expected_profit,r.portfolio_raroc]);
styleHeader(stress.getRange("A6:I6")); styleBody(stress.getRangeByIndexes(6,0,sr.length,9));
stress.getRange("C7:F9").format.numberFormat = pctFmt; stress.getRange("G7:H9").format.numberFormat = moneyFmt; stress.getRange("I7:I9").format.numberFormat = pctFmt;
stress.getRange("A:I").format.columnWidth = 16; stress.getRange("A:A").format.columnWidth = 14; stress.getRange("G:H").format.columnWidth = 21;

const profit = baseSheet("Profitability", "PD–LGD–EAD Portfolio Economics", "Risk-band view of exposure, expected loss, profit, RAROC and break-even pricing");
const pr = payload.v5_profitability;
profit.getRange("A6:H6").values = [["Risk band","Applications","Mean PD","Total EAD","Expected loss","Expected profit","Mean RAROC","Break-even rate"]];
profit.getRangeByIndexes(6,0,pr.length,8).values = pr.map(r => [r.risk_band,r.applications,r.mean_pd,r.total_ead,r.expected_loss,r.expected_profit,r.mean_raroc,r.mean_break_even_rate]);
styleHeader(profit.getRange("A6:H6")); styleBody(profit.getRangeByIndexes(6,0,pr.length,8));
profit.getRange(`C7:C${6+pr.length}`).format.numberFormat = pctFmt; profit.getRange(`D7:F${6+pr.length}`).format.numberFormat = moneyFmt; profit.getRange(`G7:H${6+pr.length}`).format.numberFormat = pctFmt;
profit.getRange("A:H").format.columnWidth = 17; profit.getRange("D:F").format.columnWidth = 21;

workbook.recalculate();
const summaryCheck = await workbook.inspect({ kind: "table", range: "V5 Summary!A2:H16", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 10 });
const strategyCheck = await workbook.inspect({ kind: "table", range: "Strategy Simulator!A5:J18", include: "values,formulas", tableMaxRows: 18, tableMaxCols: 12 });
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
await fs.writeFile(`${outputPath}.inspect.ndjson`, `${summaryCheck.ndjson}\n${strategyCheck.ndjson}\n${errors.ndjson}\n`);
for (const sheetName of ["V5 Summary","Walk Forward","Strategy Simulator","Stress Test","Profitability"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${sheetName.toLowerCase().replaceAll(" ","_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(JSON.stringify({ outputPath, sheetsAdded: 5, errorScan: errors.ndjson }));
