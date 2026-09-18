/* Leyenda persistente de las 5 clases de terreno (siempre visible junto a
   overlays y barras, nunca solo en hover). */
import { TERRAIN } from '@/lib/vocab'

export function TerrainLegend({ compact = false }: { compact?: boolean }) {
  return (
    <ul className={`flex flex-wrap items-center gap-x-3 gap-y-1 ${compact ? 'text-[10px]' : 'text-[11px]'} text-muted-foreground`}>
      {TERRAIN.map(t => (
        <li key={t.key} className="flex items-center gap-1.5">
          <span className="size-2 rounded-[3px]" style={{ background: t.hex }} />
          {t.label}
        </li>
      ))}
    </ul>
  )
}
