import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export const runtime = "nodejs";

export async function GET() {
  try {
    const repoRoot = path.resolve(process.cwd(), "..");
    const filePath = path.join(repoRoot, "data", "results", "statistics", "change_statistics.csv");

    const exists = await fs
      .access(filePath)
      .then(() => true)
      .catch(() => false);

    if (!exists) {
      return NextResponse.json({ error: "No statistics found" }, { status: 404 });
    }

    const content = await fs.readFile(filePath, "utf-8");
    const [headerLine, dataLine] = content.trim().split(/\r?\n/);

    if (!headerLine || !dataLine) {
      return NextResponse.json({ error: "Invalid CSV format" }, { status: 400 });
    }

    const headers = headerLine.split(",");
    const values = dataLine.split(",");
    const stats = headers.reduce<Record<string, string>>((acc, header, idx) => {
      acc[header] = values[idx] ?? "";
      return acc;
    }, {});

    return NextResponse.json(stats);
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || "Server error" }, { status: 500 });
  }
}
