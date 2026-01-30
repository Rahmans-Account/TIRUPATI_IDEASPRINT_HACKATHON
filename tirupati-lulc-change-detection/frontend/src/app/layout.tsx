import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Map, LayoutDashboard, Binary, BarChart3, Download, Cpu, UploadCloud, Images } from "lucide-react";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "UrbanLens | Tirupati LULC Intelligence",
  description: "Advanced AI-powered pixel-level LULC change analytics for Tirupati District",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} min-h-screen bg-[#020617] text-slate-200 antialiased font-sans`}
      >
        {/* Glowing Background Orbs (Non-intrusive) */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10">
          <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] rounded-full bg-indigo-900/20 blur-[120px]" />
          <div className="absolute top-[20%] -right-[10%] w-[30%] h-[30%] rounded-full bg-blue-900/10 blur-[100px]" />
        </div>

        {/* Glassmorphic Navbar */}
        <header className="sticky top-0 z-50 w-full border-b border-slate-800/50 bg-slate-950/70 backdrop-blur-md">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="bg-indigo-600 p-2 rounded-lg group-hover:rotate-12 transition-transform duration-300">
                <Cpu size={22} className="text-white" />
              </div>
              <div>
                <p className="text-[10px] font-black uppercase tracking-[0.3em] text-indigo-400 leading-none">
                  UrbanLens AI
                </p>
                <h1 className="text-lg font-bold tracking-tight text-white">Tirupati Intelligence</h1>
              </div>
            </Link>

            <nav className="hidden md:flex items-center gap-1 bg-slate-900/50 p-1 rounded-xl border border-slate-800">
              <NavLink href="/" icon={<LayoutDashboard size={16} />} label="Overview" />
              <NavLink href="/lulc" icon={<Map size={16} />} label="LULC Maps" />
              <NavLink href="/change" icon={<Binary size={16} />} label="Change" />
              <NavLink href="/analytics" icon={<BarChart3 size={16} />} label="Analytics" />
              <NavLink href="/gallery" icon={<Images size={16} />} label="Gallery" />
              <NavLink href="/export" icon={<Download size={16} />} label="Export" />
              <NavLink href="/upload" icon={<UploadCloud size={16} />} label="Upload" />
            </nav>

            <div className="flex items-center gap-4">
              <div className="hidden lg:flex flex-col items-end">
                <span className="text-[10px] font-mono text-slate-500">SYSTEM STATUS</span>
                <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                   <span className="h-1 w-1 bg-emerald-400 rounded-full animate-pulse" /> ONLINE
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="mx-auto w-full max-w-7xl px-6 py-8">
          <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
            {children}
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-900 bg-slate-950/50 py-8">
           <div className="mx-auto max-w-7xl px-6 flex flex-col md:flex-row justify-between items-center gap-4 text-slate-500 text-xs font-mono">
                <p>© 2026 UrbanLens | Built for Sustainable Governance</p>
                <div className="flex gap-6">
                  <span>LANDSAT 8/9 (30m)</span>
                  <span>GOOGLE EARTH ENGINE</span>
                  <span>NEXT.JS 16</span>
                </div>
           </div>
        </footer>
      </body>
    </html>
  );
}

// Custom NavLink Component for cleaner code and hover effects
function NavLink({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
  return (
    <Link
      href={href}
      className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold text-slate-400 hover:text-white hover:bg-slate-800 transition-all active:scale-95"
    >
      {icon}
      {label}
    </Link>
  );
}