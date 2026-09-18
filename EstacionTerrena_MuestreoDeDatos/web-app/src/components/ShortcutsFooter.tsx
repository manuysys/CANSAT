/* Dock flotante con los atajos de teclado, siempre a la vista. */
const KEYS: Array<[string, string]> = [
  ['↑↓', 'navegar'], ['Enter', 'detalle'], ['A', 'alertas'], ['/', 'buscar'],
  ['1-2-3', 'vistas'], ['E', 'exportar'], ['P', 'auto'], ['Esc', 'cerrar'],
]

export function ShortcutsFooter() {
  return (
    <footer className="shortcuts-dock no-print pointer-events-none fixed inset-x-0 bottom-3 z-30 flex justify-center px-4">
      <div className="glass pointer-events-auto flex flex-wrap items-center justify-center gap-x-5 gap-y-1 rounded-full border border-border/60 px-6 py-2 text-[11px] text-muted-foreground shadow-[0_16px_40px_-16px_rgba(0,0,0,.9)]">
        {KEYS.map(([k, label]) => (
          <span key={k} className="flex items-center gap-1.5">
            <kbd>{k}</kbd> {label}
          </span>
        ))}
        <span className="text-muted-foreground/60">Estación Terrena · datos de solo lectura</span>
      </div>
    </footer>
  )
}
