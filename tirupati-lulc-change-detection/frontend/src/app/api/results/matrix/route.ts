import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export const runtime = "nodejs";

export type CsvTable = {
  headers: string[];
  rows: { label: string; values: number[] }[];
};

function parseCsv(content: string): CsvTable {
  const lines = content.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return { headers: [], rows: [] };
  }

  const [headerLine, ...dataLines] = lines;
  const headers = headerLine.split(",").slice(1).map((h) => h.replace(/"/g, ""));

  const rows = dataLines.map((line) => {
    const parts = line.split(",").map((cell) => cell.replace(/"/g, ""));
    const [label, ...values] = parts;
    return {
      label,
      values: values.map((v) => Number(v)),
    };
  });

  return { headers, rows };
}

export async function GET() {
  try {
    const repoRoot = path.resolve(process.cwd(), "..");
    const filePath = path.join(repoRoot, "data", "results", "statistics", "transition_matrix.csv");

    const exists = await fs
      .access(filePath)
      .then(() => true)
      .catch(() => false);

    if (!exists) {
      return NextResponse.json({ error: "No matrix found" }, { status: 404 });
    }

    const content = await fs.readFile(filePath, "utf-8");
    const csvTable = parseCsv(content);

    return NextResponse.json(csvTable);
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || "Server error" }, { status: 500 });
  }
}
