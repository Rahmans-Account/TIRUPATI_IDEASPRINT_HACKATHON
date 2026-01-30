import { ArrowRight, ShieldCheck, BarChart3, Map as MapIcon, Zap, Globe } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <section className="space-y-10 animate-in fade-in duration-1000">
      {/* Hero Section: The "Why" */}
      <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-slate-900 p-8 md:p-12 shadow-2xl">
        {/* Decorative Background Glow */}
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-indigo-600/20 blur-[100px]"></div>
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-emerald-600/10 blur-[100px]"></div>

        <div className="relative z-10">
          <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-[0.2em] text-emerald-400">
            <Globe size={16} />
            Sustainable Urban Governance
          </div>
          <h1 className="mt-6 text-4xl md:text-5xl font-black tracking-tight text-white leading-tight">
            Pixel-level LULC <span className="text-indigo-400">Transition Analytics</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-400">
            Transforming raw satellite telemetry into actionable intelligence. 
            Our AI-driven engine quantifies urban expansion and environmental shifts 
            in Tirupati with military-grade <span className="text-white font-medium">confidence metrics.</span>
          </p>
          
          <div className="mt-10 flex flex-wrap gap-4">
            <Link href="/lulc" className="flex items-center gap-2 rounded-xl bg-indigo-600 px-6 py-3 font-bold text-white transition-all hover:bg-indigo-500 hover:scale-105 active:scale-95">
              Launch Dashboard <ArrowRight size={18} />
            </Link>
            <Link href="/change" className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/60 px-6 py-3 text-sm font-semibold text-slate-200 hover:border-indigo-500/40 hover:text-white transition-all">
              View Change Map
            </Link>
            <div className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/50 px-6 py-3 text-sm font-semibold text-slate-300">
              <Zap size={16} className="text-amber-400" />
              Processing: Landsat 8/9 (30m)
            </div>
          </div>
          <p className="mt-4 text-xs text-slate-500">
            Upload-by-user is a future feature scope; for now, run the pipeline from the terminal to refresh analytics.
          </p>
        </div>
      </div>

      {/* Feature Grid: The "What" */}
      <div className="grid gap-6 md:grid-cols-3">
        <FeatureCard 
          icon={<MapIcon className="text-blue-400" />}
          title="Multi-Temporal Maps"
          desc="5-class LULC classification for 2018 and 2024 at 30m resolution."
        />
        <FeatureCard 
          icon={<ShieldCheck className="text-emerald-400" />}
          title="Confidence Scoring"
          desc="Proprietary AI certainty overlays to ensure data reliability for policy makers."
        />
        <FeatureCard 
          icon={<BarChart3 className="text-indigo-400" />}
          title="Transition Metrics"
          desc="Automated pixel-wise transition matrices and precise area statistics."
        />
      </div>

      {/* Workflow Section: The "How" */}
      <div className="rounded-3xl border border-slate-800 bg-slate-900/40 p-8 backdrop-blur-sm">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <Zap size={20} className="text-indigo-400" /> System Workflow
        </h3>
        <div className="mt-8 grid gap-8 md:grid-cols-4">
          <Step num="01" title="Ingestion" desc="Fetch Landsat composites via GEE" />
          <Step num="02" title="Inference" desc="Baseline spectral index rules" />
          <Step num="03" title="Detection" desc="Compute bi-temporal change map" />
          <Step num="04" title="Analytics" desc="Generate area & matrix CSVs" />
        </div>
      </div>
    </section>
  );
}

function FeatureCard({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
  return (
    <div className="group rounded-2xl border border-slate-800 bg-slate-900/50 p-6 transition-all hover:border-indigo-500/50 hover:bg-slate-800/50">
      <div className="mb-4 inline-block rounded-xl bg-slate-800 p-3 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-lg font-bold text-white">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-slate-500">{desc}</p>
    </div>
  );
}

function Step({ num, title, desc }: { num: string, title: string, desc: string }) {
  return (
    <div className="relative">
      <span className="text-4xl font-black text-slate-800 absolute -top-4 -left-2 opacity-50 select-none">{num}</span>
      <div className="relative z-10">
        <h4 className="text-sm font-bold text-indigo-300 uppercase tracking-tighter">{title}</h4>
        <p className="mt-1 text-xs text-slate-500">{desc}</p>
      </div>
    </div>
  );
}