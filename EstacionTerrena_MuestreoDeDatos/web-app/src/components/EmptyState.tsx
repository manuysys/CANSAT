/* Estado vacío amigable: icono lucide + frase en español claro. */
import type { LucideIcon } from 'lucide-react'

interface Props { icon: LucideIcon; title: string; hint?: string; className?: string }

export function EmptyState({ icon: Icon, title, hint, className = '' }: Props) {
  return (
    <div className={`flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border/70 bg-muted/20 px-6 py-10 text-center ${className}`}>
      <Icon className="size-6 text-muted-foreground/70" strokeWidth={1.6} />
      <p className="text-sm font-medium text-foreground/90">{title}</p>
      {hint ? <p className="max-w-[46ch] text-xs leading-relaxed text-muted-foreground">{hint}</p> : null}
    </div>
  )
}
