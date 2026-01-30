"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { TrendingUp, Activity, Download, Info, AlertCircle } from "lucide-react";

type CsvTable = {
  headers: string[];
  rows: { label: string; values: number[] }[];
};

export default function AnalyticsPage() {
  const [mounted, setMounted] = useState(false);
  const [transition, setTransition] = useState<CsvTable | null>(null);
  const [stats, setStats] = useState<Record<string, string> | null>(null);
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

  const statMeta: Record<string, { label: string; unit: string; format: (v: number) => string }> = {
    total_changed_pixels: {
      label: "Total Changed Pixels",
      unit: "px",
      format: (v) => Math.round(v).toLocaleString(),
    },
    total_changed_area_sqm: {
      label: "Changed Area",
      unit: "m²",
      format: (v) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }),
    },
    total_changed_area_sqkm: {
      label: "Changed Area",
      unit: "km²",
      format: (v) => v.toLocaleString(undefined, { maximumFractionDigits: 2 }),
    },
    change_percentage: {
      label: "Change Percentage",
      unit: "%",
      format: (v) => v.toFixed(2),
    },
  };

  const changePct = stats?.change_percentage ? Number(stats.change_percentage) : null;

  // Helper to color-code values in the matrix (Heatmap effect)
  const getCellIntensity = (val: number) => {
    if (val === 0) return "text-slate-500 opacity-40";
    if (val > 100) return "bg-indigo-500/20 text-indigo-400 font-bold";
    return "text-slate-300";
  };

  return (
    <section className="space-y-8 p-1">
      {/* Header with Glassmorphism */}
      <div className="flex items-end justify-between border-b border-slate-800 pb-6">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2">
            <Activity className="text-indigo-500" /> Geospatial Insights
          </h2>
          <p className="mt-2 text-slate-400 max-w-2xl">
            Pixel-wise transition analysis and temporal growth metrics for Tirupati Region.
          </p>
        </div>
        <Link href="/export" className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-all text-sm font-medium">
          <Download size={16} /> Export Report
        </Link>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Left Column: Summary Stats Cards */}
        <div className="lg:col-span-1 space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-6">
              <TrendingUp size={18} className="text-emerald-400" /> Change Summary
            </h3>
            
            {stats ? (
              <div className="space-y-4">
                {Object.entries(stats).map(([key, value]) => {
                  const meta = statMeta[key];
                  const numericValue = Number(value);
                  const displayValue = meta ? meta.format(numericValue) : value;
                  const label = meta?.label ?? key.replaceAll("_", " ");
                  const unit = meta?.unit ?? "";
                  return (
                  <div key={key} className="group flex flex-col p-3 rounded-xl hover:bg-slate-800/50 transition-colors border border-transparent hover:border-slate-700">
                    <span className="text-xs uppercase tracking-wider text-slate-500 font-bold">
                      {label}
                    </span>
                    <span className="text-2xl font-mono text-white mt-1">
                      {displayValue} {unit && <span className="text-sm text-slate-500">{unit}</span>}
                    </span>
                  </div>
                );
                })}
              </div>
            ) : (
              <EmptyState message="No change statistics found." />
            )}
          </div>
          
          {/* AI Insight Auto-Generator (The "Wow" Factor) */}
          <div className="rounded-2xl bg-gradient-to-br from-indigo-900/40 to-slate-900 border border-indigo-500/30 p-6">
             <div className="flex items-center gap-2 text-indigo-300 font-semibold mb-2 text-sm">
                <Info size={16} /> AI Logic Insights
             </div>
             <p className="text-slate-300 text-sm italic leading-relaxed">
                {changePct !== null
                  ? `Overall change detected: ${changePct.toFixed(2)}%. Review transition matrix for class-level shifts.`
                  : "Run analysis to generate change insights and class-level shifts."}
             </p>
          </div>
        </div>

        {/* Right Column: Heatmap Transition Matrix */}
        <div className="lg:col-span-2 rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-semibold text-white">Transition Matrix (Pixel Shift)</h3>
            <span className="text-[10px] text-slate-500 bg-slate-800 px-2 py-1 rounded">UNITS: PIXELS</span>
          </div>

          {transition ? (
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="bg-slate-800/50">
                    <th className="px-4 py-3 text-left font-bold text-indigo-400 uppercase tracking-tighter border-b border-slate-700">
                      FROM \ TO
                    </th>
                    {transition.headers.map((header) => (
                      <th key={header} className="px-4 py-3 text-left font-semibold text-slate-300 border-b border-slate-700">
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {transition.rows.map((row) => (
                    <tr key={row.label} className="hover:bg-white/5 transition-colors">
                      <td className="px-4 py-4 font-bold text-slate-400 bg-slate-800/20">{row.label}</td>
                      {row.values.map((value: number, idx) => (
                        <td 
                          key={`${row.label}-${idx}`} 
                          className={`px-4 py-4 font-mono transition-all ${getCellIntensity(value)}`}
                        >
                          {value.toLocaleString()}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState message="No transition matrix found." />
          )}
        </div>
      </div>
    </section>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-10 text-center space-y-3">
      <AlertCircle className="text-slate-700" size={40} />
      <p className="text-slate-500 text-sm">{message}</p>
    </div>
  );
}