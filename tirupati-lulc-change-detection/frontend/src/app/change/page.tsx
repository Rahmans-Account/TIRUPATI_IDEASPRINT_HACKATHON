"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { getPublicResultImage, hasPublicResultImage } from "@/lib/results";
import { Map, Layers, ShieldCheck, Maximize2, Info, Eye } from "lucide-react";

const changeImages = [
  {
    key: "change_map.png",
    title: "Change Map",
    desc: "Binary overlay showing pixels where land use shifted.",
    icon: <Map className="text-rose-500" size={18} />
  },
  {
    key: "transition_map.png",
    title: "Transition Map",
    desc: "Categorical shifts (e.g., Forest to Urban) color-coded.",
    icon: <Layers className="text-amber-500" size={18} />
  },
  {
    key: "change_confidence.png",
    title: "Change Confidence",
    desc: "Probability map showing model certainty per pixel.",
    icon: <ShieldCheck className="text-emerald-500" size={18} />
  },
];

export default function ChangePage() {
  const [mounted, setMounted] = useState(false);
  // Track image errors for each change image key
  const [imgErrors, setImgErrors] = useState<{ [key: string]: boolean }>({});

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <section className="space-y-8 animate-in fade-in duration-700">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white">Visual Delta Analysis</h2>
          <p className="mt-2 text-slate-400">
            Bi-temporal change detection derived from Landsat composites and pixel-level classification.
          </p>
        </div>
        
        <div className="flex gap-2">
           <div className="bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-xs font-mono text-slate-300 tracking-wider uppercase">Inference: Complete</span>
           </div>
        </div>
      </div>

      {/* Grid Layout */}
      <div className="grid gap-8 lg:grid-cols-2">
        {changeImages.map((image) => {
          const imgError = imgErrors[image.key];
          return (
            <div 
              key={image.key} 
              className="group relative flex flex-col rounded-3xl border border-slate-800 bg-slate-900/40 shadow-2xl transition-all hover:border-indigo-500/50 hover:bg-slate-900/60"
            >
              {/* Card Header */}
              <div className="p-5 flex items-start justify-between">
                <div className="flex gap-4">
                  <div className="p-3 bg-slate-800 rounded-2xl group-hover:bg-indigo-500/10 transition-colors">
                    {image.icon}
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white tracking-tight">{image.title}</h3>
                    <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-semibold">{image.desc}</p>
                  </div>
                </div>
                <button className="p-2 text-slate-500 hover:text-white transition-colors" title="Expand image">
                  <Maximize2 size={18} />
                </button>
              </div>

              {/* Image Container */}
              <div className="relative aspect-square px-5 pb-5 overflow-hidden">
                {!imgError ? (
                  <div className="relative h-full w-full overflow-hidden rounded-2xl border border-slate-700 shadow-inner group-hover:border-indigo-500/30 transition-all">
                    {/* Hover Overlay */}
                    <div className="absolute inset-0 z-10 bg-indigo-900/0 opacity-0 group-hover:bg-indigo-900/10 group-hover:opacity-100 transition-all flex items-center justify-center pointer-events-none">
                       <span className="flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full border border-white/20 text-white text-sm font-medium">
                         <Eye size={16} /> View Detail
                       </span>
                    </div>

                    <Image
                      src={getPublicResultImage(image.key)}
                      alt={image.title}
                      fill
                      className="object-cover transition-transform duration-500 group-hover:scale-105"
                      onError={() => setImgErrors((prev) => ({ ...prev, [image.key]: true }))}
                    />
                  </div>
                ) : (
                  <div className="flex h-full w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-800 bg-slate-950/50 p-6 text-center">
                    <div className="mb-4 rounded-full bg-slate-900 p-4">
                      <Info className="text-slate-600" size={32} />
                    </div>
                    <p className="text-sm font-medium text-slate-400">Data Node Missing</p>
                    <p className="mt-1 text-xs text-slate-600">Run export script for {image.key}</p>
                  </div>
                )}
              </div>
              
              {/* Interactive Tooltip / Legend Preview */}
              <div className="px-5 pb-5">
                <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                   <div
                     className={`h-full bg-indigo-500 transition-all duration-1000 ${!imgError ? 'w-full' : 'w-0'}`}
                   ></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}