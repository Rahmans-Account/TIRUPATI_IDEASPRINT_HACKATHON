"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Download, FileText, ArrowLeft, BarChart3 } from "lucide-react";

type CsvTable = {
  headers: string[];
  rows: { label: string; values: number[] }[];
};

export default function ReportPage() {
  const [mounted, setMounted] = useState(false);
  const [stats, setStats] = useState<Record<string, string> | null>(null);
  const [transition, setTransition] = useState<CsvTable | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setMounted(true);
    fetchData();
  }, []);

  async function fetchData() {
    try {
      const [statsRes, matrixRes] = await Promise.all([
        fetch("/api/results/stats"),
        fetch("/api/results/matrix"),
      ]);

      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }

      if (matrixRes.ok) {
        const matrixData = await matrixRes.json();
        setTransition(matrixData);
      }
    } catch (err) {
      console.error("Failed to fetch results:", err);
    } finally {
      setLoading(false);
    }
  }

  if (!mounted || loading) return null;

  const changePct = stats?.change_percentage ? Number(stats.change_percentage) : null;
  const areaSqKm = stats?.total_changed_area_sqkm ? Number(stats.total_changed_area_sqkm) : null;
  const pixels = stats?.total_changed_pixels ? Number(stats.total_changed_pixels) : null;

  return (
    <section className="space-y-8">
      <div className="flex items-center justify-between border-b border-slate-800 pb-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-indigo-400 text-sm font-semibold uppercase tracking-widest">
            <FileText size={14} /> Executive Summary
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white">Tirupati LULC Change Report</h2>
          <p className="text-slate-400 max-w-2xl">
            Auto-compiled summary from the latest classification and change detection outputs.
          </p>
        </div>
        <Link
          href="/export"
          className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 hover:text-white hover:border-indigo-500/50 transition-all"
        >
          <ArrowLeft size={16} /> Back to Export
        </Link>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <BarChart3 size={18} className="text-indigo-400" /> Key Findings
            </h3>
            {stats ? (
              <ul className="mt-4 space-y-2 text-sm text-slate-300">
                <li>
                  • Overall change detected: {changePct !== null ? changePct.toFixed(2) : "N/A"}%
                </li>
                <li>
                  • Changed area: {areaSqKm !== null ? areaSqKm.toFixed(2) : "N/A"} km²
                </li>
                <li>
                  • Changed pixels: {pixels !== null ? Math.round(pixels).toLocaleString() : "N/A"}
                </li>
                <li>• Data source: Landsat 8/9 composites (30m)</li>
                <li>• Method: Baseline spectral-index classifier</li>
              </ul>
            ) : (
              <p className="mt-4 text-sm text-slate-500">No statistics found. Run the pipeline first.</p>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
            <h3 className="text-lg font-semibold text-white">Transition Matrix Summary</h3>
            {transition ? (
              <div className="mt-4 overflow-x-auto rounded-xl border border-slate-800">
                <table className="min-w-full text-xs">
                  <thead>
                    <tr className="bg-slate-800/50">
                      <th className="px-3 py-2 text-left font-semibold text-slate-300">FROM \ TO</th>
                      {transition.headers.map((header) => (
                        <th key={header} className="px-3 py-2 text-left text-slate-400">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {transition.rows.map((row) => (
                      <tr key={row.label}>
                        <td className="px-3 py-2 font-semibold text-slate-400 bg-slate-800/20">
                          {row.label}
                        </td>
                        {row.values.map((value, idx) => (
                          <td key={`${row.label}-${idx}`} className="px-3 py-2 text-slate-300">
                            {value.toLocaleString()}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="mt-4 text-sm text-slate-500">No transition matrix found.</p>
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
            <h3 className="text-lg font-semibold text-white">Downloads</h3>
            <div className="mt-4 flex flex-col gap-3">
              <a
                className="flex items-center gap-2 rounded-lg bg-slate-800 px-4 py-2 text-sm text-slate-200 hover:bg-indigo-600 transition-all"
                href="/results/transition_matrix.csv"
                download
              >
                <Download size={16} /> Transition Matrix (CSV)
              </a>
              <a
                className="flex items-center gap-2 rounded-lg bg-slate-800 px-4 py-2 text-sm text-slate-200 hover:bg-indigo-600 transition-all"
                href="/results/change_statistics.csv"
                download
              >
                <Download size={16} /> Change Statistics (CSV)
              </a>
            </div>
          </div>

          <div className="rounded-2xl border border-indigo-500/30 bg-indigo-950/30 p-6">
            <h3 className="text-sm font-semibold text-indigo-300">Accuracy Notice</h3>
            <p className="mt-2 text-xs text-slate-400">
              This report is generated from a baseline spectral-index classifier. It is not a validated
              accuracy assessment. For accuracy metrics, add validation samples and compute confusion
              matrices.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
