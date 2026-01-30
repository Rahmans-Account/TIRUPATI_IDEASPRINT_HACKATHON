import Link from "next/link";
import { UploadCloud, FileImage, Calendar, ShieldCheck, Zap, Database, Info } from "lucide-react";

export default function UploadPage() {
  return (
    <section className="space-y-8">
      <div className="flex items-center justify-between border-b border-slate-800 pb-6">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/30 bg-indigo-500/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-indigo-300">
            <ShieldCheck size={12} /> Future Scope Demo
          </div>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-white">Upload & Analyze</h2>
          <p className="mt-2 text-slate-400 max-w-2xl">
            This interface demonstrates the planned upload workflow. For now, generate analytics from the terminal
            and refresh the dashboard to view results.
          </p>
        </div>
        <Link href="/" className="text-sm text-slate-400 hover:text-white transition-colors">
          Back to Overview
        </Link>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-8">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-indigo-500/15 p-3 text-indigo-300">
              <UploadCloud size={20} />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">Upload two Landsat GeoTIFFs</h3>
              <p className="text-xs text-slate-400">Demo-only UI (uploads disabled)</p>
            </div>
          </div>

          <div className="mt-6 grid gap-6 md:grid-cols-2">
            <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5 space-y-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Calendar size={16} className="text-indigo-300" /> Year 1
              </div>
              <input
                type="number"
                placeholder="2018"
                disabled
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-4 py-2 text-slate-500"
              />
              <div className="rounded-xl border border-dashed border-slate-800 bg-slate-950/80 p-5 text-center text-slate-500">
                <FileImage className="mx-auto mb-2" size={20} />
                Drop .tif here (disabled)
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5 space-y-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Calendar size={16} className="text-indigo-300" /> Year 2
              </div>
              <input
                type="number"
                placeholder="2024"
                disabled
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-4 py-2 text-slate-500"
              />
              <div className="rounded-xl border border-dashed border-slate-800 bg-slate-950/80 p-5 text-center text-slate-500">
                <FileImage className="mx-auto mb-2" size={20} />
                Drop .tif here (disabled)
              </div>
            </div>
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              disabled
              className="inline-flex items-center gap-2 rounded-xl bg-indigo-600/50 px-5 py-2.5 text-sm font-semibold text-white/70 cursor-not-allowed"
            >
              <Zap size={16} /> Run Analysis (disabled)
            </button>
            <span className="text-xs text-slate-500">Planned: 45–70s fast pipeline</span>
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Database size={16} className="text-emerald-400" /> What will happen after upload
            </h4>
            <ol className="mt-4 space-y-2 text-xs text-slate-400 list-decimal list-inside">
              <li>Files stored in data/raw/landsat/&lt;year&gt;/</li>
              <li>Clip rasters to Tirupati boundary</li>
              <li>Baseline LULC classification</li>
              <li>Change detection + transition matrix</li>
              <li>Analytics + export visuals</li>
            </ol>
          </div>

          <div className="rounded-2xl border border-indigo-500/30 bg-indigo-950/30 p-6">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Info size={16} className="text-indigo-300" /> Run from terminal (current)
            </h4>
            <p className="mt-2 text-xs text-slate-400">
              Use the fast pipeline to generate analytics, then refresh the dashboard pages.
            </p>
            <ul className="mt-3 text-xs text-slate-400 space-y-1 list-disc list-inside">
              <li>Outputs: data/results and frontend/public/results</li>
              <li>Refresh pages after the run completes</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
