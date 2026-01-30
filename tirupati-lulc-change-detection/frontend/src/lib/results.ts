// Utility functions for frontend image and result handling
// All file I/O moved to API routes for server-side operations

export type CsvTable = {
  headers: string[];
  rows: { label: string; values: number[] }[];
};

/**
 * Get the public URL path for a result image
 * @param imageName - Name of the image file (e.g., 'lulc_2018.png')
 * @returns Public URL path
 */

export function getPublicResultImage(imageName: string): string {
  return `/results/maps/${imageName}`;
}

/**
 * Check if a result image exists (client-side assumption)
 * Returns true to allow Image component to attempt loading
 * Fallback to empty state if 404
 */
export function hasPublicResultImage(imageName: string): boolean {
  // Client-side: assume files may exist, let Next.js Image handle 404
  return true;
}
