import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import { spawn } from "child_process";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function saveFile(file: File, targetPath: string) {
  const arrayBuffer = await file.arrayBuffer();
  await fs.writeFile(targetPath, Buffer.from(arrayBuffer));
}

function runCommand(cmd: string, args: string[], cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, stdio: "pipe" });
    let stderr = "";

    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    child.on("close", (code) => {
      if (code === 0) return resolve();
      reject(new Error(stderr || `Command failed: ${cmd} ${args.join(" ")}`));
    });
  });
}

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const year1 = form.get("year1")?.toString();
    const year2 = form.get("year2")?.toString();
    const file1 = form.get("file1");
    const file2 = form.get("file2");

    if (!year1 || !year2) {
      return NextResponse.json({ error: "Missing years" }, { status: 400 });
    }

    if (!(file1 instanceof File) || !(file2 instanceof File)) {
      return NextResponse.json({ error: "Missing files" }, { status: 400 });
    }

    const repoRoot = path.resolve(process.cwd(), "..");
    const landsatRoot = path.join(repoRoot, "data", "raw", "landsat");

    const year1Dir = path.join(landsatRoot, year1);
    const year2Dir = path.join(landsatRoot, year2);
    await fs.mkdir(year1Dir, { recursive: true });
    await fs.mkdir(year2Dir, { recursive: true });

    const target1 = path.join(year1Dir, `Tirupati_Landsat_${year1}.tif`);
    const target2 = path.join(year2Dir, `Tirupati_Landsat_${year2}.tif`);

    await saveFile(file1, target1);
    await saveFile(file2, target2);

    const python =
      process.env.PYTHON_EXECUTABLE ||
      "C:/Projects/hack/.venv/Scripts/python.exe";

    await runCommand(python, ["scripts/fast_pipeline.py", "--years", year1, year2], repoRoot);

    return NextResponse.json({ 
      ok: true,
      year1,
      year2,
      timestamp: new Date().toISOString()
    });
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || "Server error" }, { status: 500 });
  }
}
