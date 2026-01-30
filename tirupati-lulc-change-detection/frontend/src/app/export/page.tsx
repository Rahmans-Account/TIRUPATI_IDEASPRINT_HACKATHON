"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Download, FileSpreadsheet, Image as ImageIcon, CheckCircle2, ArrowRight } from "lucide-react";

const exports = [
  { name: "Transition Matrix", path: "/results/transition_matrix.csv", type: "CSV", size: "~1 KB" },
  { name: "Change Statistics", path: "/results/change_statistics.csv", type: "CSV", size: "~1 KB" },
  { name: "LULC 2018 Map", path: "/results/lulc_2018.png", type: "PNG", size: "~3.3 MB" },
  { name: "LULC 2024 Map", path: "/results/lulc_2024.png", type: "PNG", size: "~3.1 MB" },
  { name: "Change Map", path: "/results/change_map.png", type: "PNG", size: "~0.5 MB" },
  { name: "Transition Map", path: "/results/transition_map.png", type: "PNG", size: "~0.8 MB" },
  { name: "Change Confidence", path: "/results/change_confidence.png", type: "PNG", size: "~0.9 MB" },
  { name: "Confidence 2018", path: "/results/confidence_2018.png", type: "PNG", size: "~0.9 MB" },
  { name: "Confidence 2024", path: "/results/confidence_2024.png", type: "PNG", size: "~0.9 MB" },
];

export default function ExportPage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <section className="space-y-8 animate-in slide-in-from-bottom-4 duration-700">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white">Export Center</h2>
          <p className="mt-2 text-slate-400">
            Generate and download high-resolution assets for policy reports and urban planning.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-emerald-400 bg-emerald-500/10 px-3 py-1.5 rounded-full border border-emerald-500/20">
          <CheckCircle2 size={14} />
          ALL ASSETS VERIFIED
        </div>
      </div>

      {/* Main Export Grid */}
      <div className="grid gap-4">
        {exports.map((item, index) => (
          <div 
            key={item.path} 
            style={{ animationDelay: `${index * 100}ms` }}
            className="group flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-900/40 p-5 hover:bg-slate-800/60 hover:border-indigo-500/40 transition-all animate-in fade-in"
          >
            <div className="flex items-center gap-5">
              {/* Icon Type Discernment */}
              <div className={`p-4 rounded-xl ${
                item.type === 'CSV' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-blue-500/10 text-blue-400'
              }`}>
                {item.type === 'CSV' ? <FileSpreadsheet size={24} /> : <ImageIcon size={24} />}
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white group-hover:text-indigo-300 transition-colors">
                  {item.name}
                </h3>
                <div className="flex items-center gap-3 mt-1">
                  <span className="text-xs font-mono text-slate-500 bg-slate-800 px-2 py-0.5 rounded uppercase tracking-wider">
                    {item.type}
                  </span>
                  <span className="text-xs text-slate-600 font-medium">
                    {item.size}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <a
                href={item.path}
                download
                className="flex items-center gap-2 rounded-xl bg-slate-800 px-5 py-2.5 text-sm font-bold text-white hover:bg-indigo-600 transition-all shadow-lg active:scale-95"
              >
                <Download size={16} />
                Download
              </a>
            </div>
          </div>
        ))}
      </div>

      {/* "Ready for Policy" Callout */}
      <div className="rounded-3xl bg-gradient-to-r from-indigo-600/20 to-transparent border border-indigo-500/30 p-8 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="max-w-md">
          <h4 className="text-xl font-bold text-white mb-2">Generate Executive Summary</h4>
          <p className="text-slate-400 text-sm leading-relaxed">
            Need a consolidated PDF report for stakeholders? Our AI can wrap these statistics into a formatted policy brief.
          </p>
        </div>
        <Link
          href="/report"
          className="whitespace-nowrap flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-3 rounded-xl font-bold transition-all transform hover:translate-x-1"
        >
          Compile Full Report <ArrowRight size={18} />
        </Link>
      </div>
    </section>
  );
}