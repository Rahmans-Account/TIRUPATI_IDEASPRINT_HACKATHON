"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { getPublicResultImage, hasPublicResultImage } from "@/lib/results";
import { Calendar, Info, MapPin, Maximize2 } from "lucide-react";

// Standard LULC Legend - This makes you look like a Pro
const legend = [
  { label: "Forest", color: "bg-emerald-600" },
  { label: "Water", color: "bg-blue-500" },
  { label: "Agriculture", color: "bg-yellow-400" },
  { label: "Barren", color: "bg-orange-200" },
  { label: "Built-up", color: "bg-rose-500" },
];

export default function LulcMapsPage() {
  const [years, setYears] = useState<number[]>([2018, 2024]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Read uploaded years from localStorage
    const lastUpload = localStorage.getItem("lastUpload");
    if (lastUpload) {
      try {
        const { year1, year2 } = JSON.parse(lastUpload);
        setYears([parseInt(year1), parseInt(year2)]);
      } catch (e) {
        console.error("Failed to parse lastUpload:", e);
      }
    }
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <section className="space-y-8 animate-in fade-in slide-in-from-top-4 duration-700">
      {/* Header with Location Metadata */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-8">
        <div>
          <div className="flex items-center gap-2 text-indigo-400 text-sm font-semibold mb-1 uppercase tracking-widest">
            <MapPin size={14} /> Region: Tirupati District
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white">LULC Classification Maps</h2>
          <p className="mt-2 text-slate-400 max-w-2xl">
            Comparative analysis of Land Use and Land Cover (LULC) shifts using Landsat 8/9 composites over a 6-year window.
          </p>
        </div>
        
        {/* Dynamic Legend Component */}
        <div className="bg-slate-900/80 border border-slate-700 p-4 rounded-2xl backdrop-blur-md">
          <p className="text-[10px] font-bold text-slate-500 uppercase mb-3 tracking-widest text-center">Map Legend</p>
          <div className="flex flex-wrap gap-4 justify-center">
            {legend.map((item) => (
              <div key={item.label} className="flex items-center gap-2">
                <span className={`h-3 w-3 rounded-full ${item.color} shadow-[0_0_8px_rgba(0,0,0,0.5)]`}></span>
                <span className="text-xs font-medium text-slate-300">{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Map Grid */}
      <div className="grid gap-8 md:grid-cols-2">
        {years.map((year, index) => {
          const imageName = `lulc_${year}.png`;
          const hasImage = hasPublicResultImage(imageName);
          return (
            <div 
              key={year} 
              className="group relative flex flex-col rounded-3xl border border-slate-800 bg-slate-900/40 shadow-2xl transition-all hover:border-indigo-500/50"
            >
              {/* Year Badge */}
              <div className="absolute top-6 left-6 z-20">
                <div className="flex items-center gap-2 bg-black/60 backdrop-blur-md border border-white/10 px-4 py-2 rounded-xl shadow-2xl">
                  <Calendar size={16} className="text-indigo-400" />
                  <span className="text-xl font-black text-white tracking-tighter">YEAR {year}</span>
                </div>
              </div>

              {/* Action Toolbar */}
              <div className="absolute top-6 right-6 z-20 opacity-0 group-hover:opacity-100 transition-opacity">
                 <button className="bg-white/10 backdrop-blur-md p-2 rounded-lg border border-white/20 text-white hover:bg-white/20">
                    <Maximize2 size={18} />
                 </button>
              </div>

              {/* Map Canvas */}
              <div className="relative aspect-square p-3">
                {hasImage ? (
                  <div className="relative h-full w-full overflow-hidden rounded-2xl border border-slate-700">
                    <Image
                      src={getPublicResultImage(imageName)}
                      alt={`LULC ${year}`}
                      fill
                      className="object-cover transition-transform duration-700 group-hover:scale-110"
                    />
                    {/* Corner coordinates label - adds "Realism" */}
                    <div className="absolute bottom-4 right-4 text-[10px] font-mono text-white/40 pointer-events-none">
                      Lat: 13.6288° N, Lon: 79.4192° E
                    </div>
                  </div>
                ) : (
                  <div className="flex h-full w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-800 bg-slate-950/50 p-6 text-center">
                    <div className="mb-4 rounded-full bg-slate-900 p-4">
                      <Info className="text-slate-600" size={32} />
                    </div>
                    <p className="text-sm font-medium text-slate-400 tracking-tight">Image Data Missing</p>
                    <p className="mt-1 text-xs text-slate-600 italic">Expected: {imageName}</p>
                  </div>
                )}
              </div>
              
              {/* Brief Labeling at bottom */}
              <div className="px-6 pb-6 text-center">
                 <p className="text-xs font-mono text-slate-500 uppercase tracking-widest">
                   Satellite: Landsat 8/9 | Resolution: 30m/px
                 </p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}