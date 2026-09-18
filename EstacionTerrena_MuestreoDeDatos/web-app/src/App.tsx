/* Estación Terrena LB135 — composición de vistas con crossfade de 200 ms,
   fondo ambiental, cutscene de arranque y hero 3D en la vista Vuelo. */
import { lazy, Suspense, useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Backdrop } from '@/components/Backdrop'
import { Cutscene, cutscenePending } from '@/components/Cutscene'
import { TopBar } from '@/components/TopBar'
import { Narrative } from '@/components/Narrative'
import { Kpis } from '@/components/Kpis'
import { DamageChart } from '@/components/DamageChart'
import { Corridor } from '@/components/Corridor'

/* three.js vive en un chunk async: el dashboard pinta antes que el 3D. */
const DescentScene = lazy(() =>
  import('@/components/scene/DescentScene').then(m => ({ default: m.DescentScene })),
)

function SceneFallback() {
  return (
    <div className="flex h-[402px] animate-pulse flex-col overflow-hidden rounded-xl border border-border/60 bg-card/40 md:h-[462px]">
      <div className="border-b border-border/50 px-5 py-2.5">
        <div className="h-3 w-48 rounded bg-muted/40" />
      </div>
      <div className="grid flex-1 place-items-center font-mono text-[10.5px] text-muted-foreground/60">
        cargando motor 3d…
      </div>
    </div>
  )
}
import { Detail } from '@/components/Detail'
import { Alerts } from '@/components/Alerts'
import { Timeline } from '@/components/Timeline'
import { Toasts } from '@/components/Toasts'
import { Jurado } from '@/components/Jurado'
import { Scrubber } from '@/components/Scrubber'
import { AltProfile } from '@/components/AltProfile'
import { Bitacora } from '@/components/Bitacora'
import { GpsTrack } from '@/components/GpsTrack'
import { Sampling } from '@/components/Sampling'

const PostView = lazy(() => import('@/components/PostView').then(m => ({ default: m.PostView })))
const ReportView = lazy(() => import('@/components/ReportView').then(m => ({ default: m.ReportView })))
import { Presentation } from '@/components/Presentation'
import { ShortcutsFooter } from '@/components/ShortcutsFooter'
import { useFeed } from '@/hooks/useFeed'
import { useKeyboard } from '@/hooks/useKeyboard'
import { startTour, tourDone } from '@/lib/tour'
import { useMission } from '@/store/mission'

/** Arranca el tour una sola vez, cuando ya hay datos pintados en pantalla y
    la cutscene de arranque terminó (si se reproduce). */
function TourBoot() {
  const status = useMission(s => s.status)
  useEffect(() => {
    if (status !== 'live' || tourDone()) return
    let timer: number | undefined
    const arm = () => { timer = window.setTimeout(() => startTour(), 900) }
    if (cutscenePending()) {
      const onDone = () => arm()
      window.addEventListener('lb135-cutscene-done', onDone, { once: true })
      return () => {
        window.removeEventListener('lb135-cutscene-done', onDone)
        window.clearTimeout(timer)
      }
    }
    arm()
    return () => window.clearTimeout(timer)
  }, [status])
  return null
}

function VueloView() {
  const [sceneReady, setSceneReady] = useState(() => !cutscenePending())
  useEffect(() => {
    if (sceneReady) return
    const onDone = () => setSceneReady(true)
    window.addEventListener('lb135-cutscene-done', onDone, { once: true })
    return () => window.removeEventListener('lb135-cutscene-done', onDone)
  }, [sceneReady])

  return (
    <div className="mission-page flex flex-col gap-5">
      <div className="mission-hero grid gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(390px,.75fr)]">
        {sceneReady ? (
          <Suspense fallback={<SceneFallback />}>
            <DescentScene />
          </Suspense>
        ) : (
          <SceneFallback />
        )}
        <div className="flex min-w-0 flex-col gap-4">
          <Narrative />
          <Kpis className="grid-cols-2 md:grid-cols-3 xl:grid-cols-2" />
        </div>
      </div>
      <div className="mission-section-heading">
        <div><span>02 / timeline</span><h2>La historia del descenso</h2></div>
        <p>Daño estimado por altitud y momento de captura · perfil físico gemelo</p>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <DamageChart />
        <AltProfile />
      </div>
      <Scrubber />
      <div className="mission-section-heading mt-2">
        <div><span>03 / muestreo y posición</span><h2>Muestreo adaptativo y trayectoria</h2></div>
        <p>Qué decidió el sampler a bordo, con qué señal, y dónde pasó cada frame</p>
      </div>
      <Sampling />
      <GpsTrack />
      <div className="mission-section-heading mt-2">
        <div><span>03 / evidencia</span><h2>Explorador de misión</h2></div>
        <p>Selecciona un frame para abrir toda su evidencia</p>
      </div>
      <div className="mission-explorer grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="h-[480px] min-h-0"><Corridor /></div>
        <div className="h-[480px] min-h-0"><Alerts /></div>
      </div>
      <div className="mission-detail mt-4 min-w-0"><Detail /></div>
      <div className="mt-4"><Bitacora /></div>
      <div className="mission-section-heading mt-2">
        <div><span>04 / registro</span><h2>Timeline de telemetría</h2></div>
        <p>Filtra, ordena y exporta el contrato de misión</p>
      </div>
      <Timeline />
    </div>
  )
}

export default function App() {
  useFeed()
  useKeyboard()
  const view = useMission(s => s.view)

  return (
    <TooltipProvider delayDuration={150}>
      <Backdrop />
      <Cutscene />
      <TopBar />
      <main id="main-content" className="mx-auto w-full max-w-[1720px] px-6 pb-24 pt-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0, y: 8, filter: 'blur(6px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            exit={{ opacity: 0, y: -6, filter: 'blur(6px)' }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            {view === 'vuelo' && <VueloView />}
            {view === 'post' && (
              <Suspense fallback={<div className="h-[60vh] animate-pulse rounded-xl bg-card/40" />}>
                <PostView />
              </Suspense>
            )}
            {view === 'informe' && (
              <Suspense fallback={<div className="h-[60vh] animate-pulse rounded-xl bg-card/40" />}>
                <ReportView />
              </Suspense>
            )}
          </motion.div>
        </AnimatePresence>
      </main>
      <ShortcutsFooter />
      <Presentation />
      <Jurado />
      <Toasts />
      <TourBoot />
    </TooltipProvider>
  )
}
