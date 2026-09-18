/* Tooltip de glosario de una línea (shadcn Tooltip) para términos técnicos. */
import type { ReactNode } from 'react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { GLOSSARY } from '@/lib/vocab'
import { normKey } from '@/lib/format'

interface Props { term: keyof typeof GLOSSARY | string; children: ReactNode }

export function Gloss({ term, children }: Props) {
  const text = GLOSSARY[normKey(term)] ?? GLOSSARY[term as string]
  if (!text) return <>{children}</>
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="cursor-help underline decoration-dotted decoration-muted-foreground/50 underline-offset-4">
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-[280px] text-xs leading-relaxed">{text}</TooltipContent>
    </Tooltip>
  )
}
