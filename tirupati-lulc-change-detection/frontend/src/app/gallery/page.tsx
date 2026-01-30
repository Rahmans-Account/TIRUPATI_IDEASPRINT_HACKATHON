
"use client";

// Data for gallery tabs
const mapsData = [
  {
    title: "LULC Map 2018",
    path: "/results/maps/lulc_2018.png",
    description: "Land Use Land Cover classification for 2018."
  },
  {
    title: "LULC Map 2024",
    path: "/results/maps/lulc_2024.png",
    description: "Land Use Land Cover classification for 2024."
  },
  {
    title: "Change Detection Enhanced",
    path: "/results/maps/change_detection_enhanced.png",
    description: "Enhanced map showing detected changes between 2018 and 2024."
  },
  {
    title: "Transition Heatmap",
    path: "/results/maps/transition_heatmap.png",
    description: "Heatmap of LULC transitions."
  },
  {
    title: "2018-2024 Comparison",
    path: "/results/maps/comparison_2018_2024.png",
    description: "Side-by-side comparison of LULC for 2018 and 2024."
  },
];

const chartsData = [
  {
    title: "Area Comparison",
    path: "/results/charts/area_comparison.png",
    description: "Area-wise comparison of LULC classes."
  },
  {
    title: "Percentage Change",
    path: "/results/charts/percentage_change.png",
    description: "Percentage change in LULC classes between years."
  },
  {
    title: "Pie Comparison",
    path: "/results/charts/pie_comparison.png",
    description: "Pie chart comparison of LULC classes."
  },
];

const interactiveData = [
  {
    title: "Interactive Comparison",
    path: "/results/interactive/interactive_comparison.html",
    description: "Interactive map for exploring LULC changes."
  },
  {
    title: "Sankey Transitions",
    path: "/results/interactive/sankey_transitions.html",
    description: "Sankey diagram of LULC transitions."
  },
];


import { useState } from "react";
import Image from "next/image";
import { Images, BarChart3, Activity, Download, Info, ExternalLink, Filter, Globe } from "lucide-react";

const tabs = [
  { id: "maps", label: "Intelligence Maps", icon: <Images size={18} /> },
  { id: "charts", label: "Growth Analytics", icon: <BarChart3 size={18} /> },
  { id: "interactive", label: "Dynamic Flows", icon: <Activity size={18} /> },
];

export default function GalleryPage() {
  const [activeTab, setActiveTab] = useState("maps");

  return (
    <section className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Header with Stats Overlays */}
      <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/40 p-10 backdrop-blur-sm">
        <div className="absolute top-0 right-0 p-8 opacity-10">
           <Globe size={120} className="text-indigo-500" />
        </div>
        
        <div className="relative z-10 text-center md:text-left">
          <h1 className="text-4xl font-black tracking-tight text-white mb-4">
            Visualization <span className="text-indigo-400">Intelligence Hub</span>
          </h1>
          <p className="text-slate-400 max-w-2xl text-lg leading-relaxed">
            A comprehensive repository of classified geospatial assets, temporal charts, and interactive transition models for the Tirupati ecosystem.
          </p>
          
          <div className="mt-8 flex flex-wrap justify-center md:justify-start gap-3">
             <StatBadge label="Cartographic" count={5} color="text-emerald-400" />
             <StatBadge label="Statistical" count={3} color="text-blue-400" />
             <StatBadge label="Simulations" count={2} color="text-purple-400" />
          </div>
        </div>
      </div>

      {/* Modern Tab Navigation */}
      <div className="flex flex-col md:flex-row justify-between items-center gap-6 sticky top-24 z-40 bg-[#020617]/80 backdrop-blur-lg py-4 border-y border-slate-800/50">
        <div className="flex bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800 shadow-inner">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-bold text-sm transition-all duration-300 ${
                activeTab === tab.id
                  ? "bg-indigo-600 text-white shadow-[0_0_20px_rgba(79,70,229,0.4)] scale-105"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
        
        <div className="flex items-center gap-2 text-xs font-mono text-slate-500 uppercase tracking-tighter">
           <Filter size={14} /> Refined by: Inference Resolution 10m
        </div>
      </div>

      {/* Dynamic Content Grid */}
      <div className="min-h-[500px]">
        {activeTab === "maps" && (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            {mapsData.map((viz, index) => (
              <VizCard key={index} viz={viz} type="map" />
            ))}
          </div>
        )}

        {activeTab === "charts" && (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            {chartsData.map((viz, index) => (
              <VizCard key={index} viz={viz} type="chart" />
            ))}
          </div>
        )}

        {activeTab === "interactive" && (
          <div className="space-y-12">
            {interactiveData.map((viz, index) => (
              <div key={index} className="group overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/60 shadow-2xl transition-all hover:border-indigo-500/30">
                <div className="p-6 border-b border-slate-800 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                  <div>
                    <h3 className="text-2xl font-bold text-white group-hover:text-indigo-400 transition-colors">{viz.title}</h3>
                    <p className="text-slate-500 text-sm mt-1">{viz.description}</p>
                  </div>
                  <a 
                    href={viz.path} target="_blank"
                    className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2 rounded-xl text-sm font-bold transition-all"
                  >
                    Full Screen <ExternalLink size={16} />
                  </a>
                </div>
                <div className="p-4 bg-black/40">
                  <iframe src={viz.path} className="h-[550px] w-full rounded-2xl border-0 shadow-inner" title={viz.title || "Visualization"} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Professional Tech Footer */}
      <div className="rounded-3xl border border-indigo-500/20 bg-indigo-500/5 p-8 text-slate-300">
        <div className="flex items-start gap-4">
           <div className="p-3 bg-indigo-500/20 rounded-2xl text-indigo-400">
              <Info size={24} />
           </div>
           <div>
              <h4 className="text-lg font-bold text-white">Advanced Visualization Pipeline</h4>
              <p className="text-sm mt-1 leading-relaxed text-slate-400">
                These visuals are synthesized using a Python-based Matplotlib & Plotly pipeline. 
                Spatial data is normalized using the WGS 84 coordinate system before rendering.
              </p>
              <div className="mt-4 flex gap-4 font-mono text-[10px] text-indigo-300/60 uppercase">
                 <span>• antialiased rendering</span>
                 <span>• categorical normalization</span>
                 <span>• pixel-wise confidence weighting</span>
              </div>
           </div>
        </div>
      </div>
    </section>
  );
}

// Sub-components for cleaner structure
function StatBadge({ label, count, color }: { label: string, count: number, color: string }) {
  return (
    <div className="bg-slate-800/50 border border-slate-700 px-4 py-2 rounded-2xl flex items-center gap-3">
      <span className={`text-xl font-black ${color}`}>{count}</span>
      <span className="text-[10px] uppercase font-bold text-slate-500 tracking-widest">{label}</span>
    </div>
  );
}

function VizCard({ viz, type }: { viz: any, type: string }) {
  return (
    <div className="group flex flex-col rounded-3xl border border-slate-800 bg-slate-900/40 transition-all hover:border-indigo-500/40 hover:bg-slate-900/60 overflow-hidden">
      <div className="relative aspect-video w-full overflow-hidden bg-slate-950">
        <Image 
          src={viz.path} alt={viz.title} fill unoptimized
          className="object-cover transition-transform duration-700 group-hover:scale-105 opacity-80 group-hover:opacity-100" 
        />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-transparent to-transparent opacity-60" />
      </div>
      <div className="p-6">
        <h3 className="text-xl font-bold text-white mb-2">{viz.title}</h3>
        <p className="text-sm text-slate-500 leading-relaxed mb-6">{viz.description}</p>
        <a 
          href={viz.path} download
          className="inline-flex items-center gap-2 rounded-xl bg-slate-800 px-5 py-2.5 text-sm font-bold text-white hover:bg-white hover:text-black transition-all"
        >
          <Download size={16} /> Asset Download
        </a>
      </div>
    </div>
  );
}