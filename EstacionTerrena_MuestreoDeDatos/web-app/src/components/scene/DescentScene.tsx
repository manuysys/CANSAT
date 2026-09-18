/* Escena 3D del descenso (three.js + @react-three/fiber).
   - CanSat detallado: paracaídas de domo con cuerdas ancladas al borde de la
     tela, cuerpo con paneles, banda emisiva, antena con baliza que titila,
     gimbal de cámara abajo y aletas; sombra proyectada y haz del corredor.
   - La altitud sigue al frame seleccionado; ▶ Descenso reproduce la misión
     entera sin robar el scroll (lib/fly) y llevando la cámara al panel.
   - ● EN VIVO: sigue el último frame que arriva del pipeline (telemetría
     GPS/IMU escribiendo telemetry.csv durante el vuelo real).
   - Sin WebGL → perfil SVG. prefers-reduced-motion → cámara quieta. */
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame } from '@react-three/fiber'
import { motion } from 'motion/react'
import { Box, Pause, Play, Radio } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Scramble } from '@/components/Scramble'
import { useMission } from '@/store/mission'
import { num } from '@/lib/format'
import { fly } from '@/lib/fly'
import { apogeo, missionPhase, tasaAnomala, verticalSpeed } from '@/lib/missionState'
import { webglOK } from '@/lib/webgl'

/* ── Geometrías ────────────────────────────────────────────────────────── */
function useTerrain() {
  return useMemo(() => {
    const g = new THREE.PlaneGeometry(74, 38, 64, 34)
    g.rotateX(-Math.PI / 2)
    const pos = g.attributes.position as THREE.BufferAttribute
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i)
      const z = pos.getZ(i)
      const y =
        Math.sin(x * 0.16) * Math.cos(z * 0.21) * 1.15 +
        Math.sin(x * 0.42 + z * 0.33) * 0.4 +
        Math.cos(z * 0.11) * 0.55
      pos.setY(i, y)
    }
    g.computeVertexNormals()
    return g
  }, [])
}

function useStars() {
  return useMemo(() => {
    const n = 460
    const arr = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      // Semilla determinista para que la escena sea pura y estable entre renders.
      const wave = (value: number) => (Math.sin(value * 12.9898) * 43758.5453) % 1
      arr[i * 3] = (wave(i + 1) - 0.5) * 95
      arr[i * 3 + 1] = 4 + Math.abs(wave(i + 17)) * 36
      arr[i * 3 + 2] = -18 - Math.abs(wave(i + 31)) * 42
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(arr, 3))
    return g
  }, [])
}

/* Cuerdas: del BORDE del domo (r=1.55, y=0) a los anclajes del cuerpo. */
function useStrings() {
  return useMemo(() => {
    const pts: number[] = []
    const N = 8
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2
      pts.push(
        Math.cos(a) * 1.53, 0.02, Math.sin(a) * 1.53,
        Math.cos(a) * 0.3, -1.52, Math.sin(a) * 0.3,
      )
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
    return new THREE.LineSegments(
      g,
      new THREE.LineBasicMaterial({ color: '#aebdcd', transparent: true, opacity: 0.65 }),
    )
  }, [])
}

/* ── CanSat detallado ──────────────────────────────────────────────────── */
function CanSat({ stowed, landed }: { stowed: boolean; landed: boolean }) {
  const beacon = useRef<THREE.MeshStandardMaterial>(null)
  const canopy = useRef<THREE.Group>(null)
  const body = useRef<THREE.Group>(null)

  useFrame((st, dt) => {
    const t = st.clock.elapsedTime
    if (beacon.current) beacon.current.emissiveIntensity = Math.sin(t * 7) > 0.2 ? 3.2 : 0.25
    if (canopy.current) {
      const targetScale = stowed ? 0.14 : 1
      const sc = THREE.MathUtils.damp(canopy.current.scale.x, targetScale, 3, dt)
      canopy.current.scale.set(sc, sc, sc)
      canopy.current.rotation.z = THREE.MathUtils.damp(canopy.current.rotation.z, landed ? 0.9 : Math.sin(t * 0.7) * 0.045, 2, dt)
      canopy.current.rotation.x = Math.cos(t * 0.55) * 0.035
      canopy.current.position.y = THREE.MathUtils.damp(canopy.current.position.y, landed ? 0.9 : 2.08, 2, dt)
    }
    if (body.current) {
      body.current.rotation.z = Math.sin(t * 0.5 + 1.2) * 0.06
      body.current.rotation.x = Math.cos(t * 0.42 + 0.6) * 0.04
    }
  })

  const strings = useStrings()
  useEffect(() => () => {
    strings.geometry.dispose()
    ;(strings.material as THREE.Material).dispose()
  }, [strings])

  return (
    <group>
      {/* ─ paracaídas: domo + borde + ventral + cuerdas ─ */}
      <group ref={canopy} position={[0, 2.08, 0]}>
        <mesh scale={[1, 0.78, 1]}>
          <sphereGeometry args={[1.55, 36, 18, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshStandardMaterial color="#e9eff7" roughness={0.55} metalness={0.05} side={THREE.DoubleSide} />
        </mesh>
        {/* gore decorativo: franja verde cada 90° */}
        {[0, 1, 2, 3].map(i => (
          <mesh key={i} rotation-y={(i / 4) * Math.PI * 2 + Math.PI / 4} scale={[1, 0.78, 1]}>
            <sphereGeometry args={[1.56, 8, 12, -0.22, 0.44, 0, Math.PI / 2]} />
            <meshStandardMaterial color="#3ddc84" roughness={0.6} side={THREE.DoubleSide} transparent opacity={0.85} />
          </mesh>
        ))}
        <mesh rotation-x={Math.PI / 2}>
          <torusGeometry args={[1.55, 0.035, 10, 48]} />
          <meshStandardMaterial color="#22303f" roughness={0.5} metalness={0.4} />
        </mesh>
        <mesh position={[0, 1.2, 0]} rotation-x={Math.PI / 2}>
          <torusGeometry args={[0.26, 0.03, 8, 24]} />
          <meshStandardMaterial color="#22303f" roughness={0.5} />
        </mesh>
        <primitive object={strings} />
      </group>

      {/* ─ cuerpo del cansat ─ */}
      <group ref={body}>
        <mesh>
          <cylinderGeometry args={[0.42, 0.46, 1.15, 28]} />
          <meshStandardMaterial color="#1b2532" metalness={0.65} roughness={0.32} />
        </mesh>
        <mesh position={[0, 0.66, 0]}>
          <cylinderGeometry args={[0.34, 0.42, 0.18, 28]} />
          <meshStandardMaterial color="#22303f" metalness={0.5} roughness={0.4} />
        </mesh>
        <mesh position={[0, 0.18, 0]}>
          <cylinderGeometry args={[0.435, 0.435, 0.14, 28]} />
          <meshStandardMaterial color="#3ddc84" emissive="#1d9b58" emissiveIntensity={1.1} roughness={0.4} />
        </mesh>
        {/* rieles/paneles laterales */}
        {[0, 1, 2, 3].map(i => {
          const a = (i / 4) * Math.PI * 2 + Math.PI / 4
          return (
            <mesh key={i} position={[Math.cos(a) * 0.46, -0.05, Math.sin(a) * 0.46]} rotation-y={-a}>
              <boxGeometry args={[0.03, 0.72, 0.18]} />
              <meshStandardMaterial color="#2c3a4b" metalness={0.5} roughness={0.45} />
            </mesh>
          )
        })}
        {/* baliza de recuperación */}
        <mesh position={[0, 0.82, 0]}>
          <cylinderGeometry args={[0.05, 0.07, 0.12, 12]} />
          <meshStandardMaterial ref={beacon} color="#3a0d12" emissive="#ff4d5e" emissiveIntensity={2} />
        </mesh>
        {/* antena de telemetría */}
        <mesh position={[0.22, 1.15, 0]} rotation-z={-0.18}>
          <cylinderGeometry args={[0.012, 0.012, 0.85, 8]} />
          <meshStandardMaterial color="#8b9aab" metalness={0.7} roughness={0.3} />
        </mesh>
        <mesh position={[0.3, 1.56, 0]}>
          <sphereGeometry args={[0.045, 12, 12]} />
          <meshStandardMaterial color="#0b0e14" emissive="#7ee8b0" emissiveIntensity={1.6} />
        </mesh>
        {/* aletas inferiores */}
        {[0, 1, 2].map(i => {
          const a = (i / 3) * Math.PI * 2
          return (
            <mesh key={i} position={[Math.cos(a) * 0.48, -0.42, Math.sin(a) * 0.48]} rotation-y={-a}>
              <boxGeometry args={[0.03, 0.46, 0.3]} />
              <meshStandardMaterial color="#22303f" roughness={0.5} metalness={0.3} />
            </mesh>
          )
        })}
        {/* gimbal de cámara */}
        <group position={[0, -0.72, 0]}>
          <mesh>
            <boxGeometry args={[0.2, 0.16, 0.2]} />
            <meshStandardMaterial color="#111823" metalness={0.6} roughness={0.35} />
          </mesh>
          <mesh position={[0, -0.12, 0.06]} rotation-x={Math.PI / 2.4}>
            <cylinderGeometry args={[0.07, 0.09, 0.12, 16]} />
            <meshStandardMaterial color="#0b0e14" emissive="#3ddc84" emissiveIntensity={2.4} />
          </mesh>
        </group>
        <pointLight position={[0, 0.5, 0]} intensity={5} distance={7} decay={2} color="#3ddc84" />
      </group>
    </group>
  )
}

/* ── Escena ────────────────────────────────────────────────────────────── */
function SceneRig({ targetY, stowed, landed, sol }: { targetY: number; stowed: boolean; landed: boolean; sol: boolean }) {
  const rig = useRef<THREE.Group>(null)
  const shadow = useRef<THREE.Mesh>(null)
  const terra = useTerrain()
  const stars = useStars()
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useEffect(() => () => {
    terra.dispose()
    stars.dispose()
  }, [terra, stars])

  useFrame((st, dt) => {
    const t = st.clock.elapsedTime
    const g = rig.current
    if (g) {
      g.position.y = THREE.MathUtils.damp(g.position.y, targetY, 1.7, dt)
      g.position.x = reduced ? 0 : Math.sin(t * 0.32) * 0.35
      g.position.z = reduced ? 0 : Math.cos(t * 0.27) * 0.25
      g.rotation.y += reduced ? 0 : dt * 0.22
      g.rotation.z = reduced ? 0 : Math.sin(t * 0.5) * 0.05
    }
    if (shadow.current && g) {
      const h = THREE.MathUtils.clamp((g.position.y - 1.4) / 8.4, 0, 1)
      shadow.current.scale.setScalar(0.55 + h * 1.6)
      ;(shadow.current.material as THREE.MeshBasicMaterial).opacity = 0.4 * (1 - h * 0.78)
      shadow.current.position.x = g.position.x
      shadow.current.position.z = g.position.z
    }
    if (!reduced) {
      st.camera.position.x = THREE.MathUtils.damp(st.camera.position.x, 8.4 + st.pointer.x * 1.5, 2, dt)
      st.camera.position.y = THREE.MathUtils.damp(st.camera.position.y, 4.6 + st.pointer.y * 0.9, 2, dt)
      st.camera.lookAt(0, 3.2, 0)
    }
  })

  return (
    <>
      <fog attach="fog" args={[sol ? '#edf1f4' : '#070a10', 22, 62]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[7, 15, 6]} intensity={1.15} color="#dfe9ff" />
      <pointLight position={[-8, 4, -6]} intensity={2.4} color="#3f8fd1" distance={40} />

      {!sol && (
        <points geometry={stars}>
          <pointsMaterial size={0.09} color="#bcd8ff" transparent opacity={0.75} depthWrite={false} sizeAttenuation />
        </points>
      )}

      <mesh geometry={terra}>
        <meshStandardMaterial color={sol ? '#c7d2da' : '#0d141d'} flatShading roughness={0.95} metalness={0.05} />
      </mesh>
      <mesh geometry={terra} position-y={0.03}>
        <meshBasicMaterial color={sol ? '#0c7a45' : '#3ddc84'} wireframe transparent opacity={sol ? 0.14 : 0.08} />
      </mesh>

      <mesh ref={shadow} rotation-x={-Math.PI / 2} position-y={0.12}>
        <circleGeometry args={[1.05, 32]} />
        <meshBasicMaterial color="#000000" transparent opacity={0.35} depthWrite={false} />
      </mesh>

      <mesh position={[0, 6.4, 0]}>
        <cylinderGeometry args={[0.014, 0.014, 11.5, 6]} />
        <meshBasicMaterial color="#3ddc84" transparent opacity={0.22} />
      </mesh>

      <group ref={rig} position={[0, 9, 0]}>
        <CanSat stowed={stowed} landed={landed} />
      </group>
    </>
  )
}

/* ── Perfil SVG de respaldo (sin WebGL) ────────────────────────────────── */
function FallbackProfile() {
  const frames = useMission(s => s.frames)
  const selected = useMission(s => s.selected)
  const select = useMission(s => s.select)
  if (!frames.length) return <div className="grid h-full place-items-center text-xs text-muted-foreground">Sin telemetría todavía.</div>
  const alts = frames.map(f => Number(f.alt_m) || 0)
  const aMax = Math.max(...alts, 1)
  const aMin = Math.min(...alts, 0)
  const pts = frames.map((f, i) => {
    const x = 8 + (i / Math.max(1, frames.length - 1)) * 84
    const y = 8 + (1 - ((Number(f.alt_m) || 0) - aMin) / Math.max(1, aMax - aMin)) * 84
    return { f, x, y }
  })
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-full w-full">
      <polyline points={pts.map(p => `${p.x},${p.y}`).join(' ')} fill="none" stroke="#3ddc84" strokeWidth={0.6} strokeDasharray="2 1.4" />
      {pts.map(p => (
        <circle
          key={p.f.src} cx={p.x} cy={p.y} r={p.f.src === selected ? 2 : 1.2}
          fill={p.f.src === selected ? '#3ddc84' : '#6b7a8c'}
          className="cursor-pointer" onClick={() => select(p.f.src)}
        />
      ))}
    </svg>
  )
}

/* ── Panel completo ────────────────────────────────────────────────────── */
export function DescentScene() {
  const frames = useMission(s => s.frames)
  const selected = useMission(s => s.selected)
  const select = useMission(s => s.select)
  const [playing, setPlaying] = useState(false)
  const [live, setLive] = useState(false)
  const ok = useMemo(() => webglOK(), [])
  const wrap = useRef<HTMLDivElement>(null)
  const [inView, setInView] = useState(true)
  const presenting = useMission(s => s.present.on)
  const jurado = useMission(s => s.jurado)
  const sol = useMission(s => s.sol)
  const summary = useMission(s => s.summary)

  useEffect(() => {
    const el = wrap.current
    if (!el || !('IntersectionObserver' in window)) return
    const io = new IntersectionObserver(en => setInView(en.some(e => e.isIntersecting)), { threshold: 0 })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  useEffect(() => () => { fly.on = false }, [])

  const alts = frames.map(f => Number(f.alt_m) || 0)
  const altMax = frames.length ? Math.max(...alts) : 250
  const altMin = frames.length ? Math.min(...alts) : 0
  const liveFrame = frames.length ? frames[frames.length - 1] : null
  const sel = frames.find(f => f.src === selected)
  const focus = playing || !live ? sel ?? null : liveFrame
  const alt = focus ? Number(focus.alt_m) || 0 : altMax
  const norm = Math.min(1, Math.max(0, (alt - altMin) / Math.max(1, altMax - altMin)))

  // ▶ Descenso: reproduce la misión sin robar el scroll de la página.
  useEffect(() => {
    if (!playing) { fly.on = false; return }
    fly.on = true
    wrap.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    const id = window.setInterval(() => {
      const st = useMission.getState()
      const list = st.frames
      if (!list.length) { setPlaying(false); return }
      const i = list.findIndex(f => f.src === st.selected)
      if (i < 0 || i >= list.length - 1) { setPlaying(false); return }
      st.select(list[i + 1].src)
    }, 700)
    return () => window.clearInterval(id)
  }, [playing])

  const ticks = [1, 0.75, 0.5, 0.25, 0].map(p => Math.round(altMin + (altMax - altMin) * p))
  const phase = missionPhase(frames, summary)
  const stowed = phase.id === 'ascenso' || phase.id === 'sin-tlm'
  const landed = phase.id === 'aterrizado' || phase.id === 'post-vuelo'
  const vs = verticalSpeed(frames)
  const apo = apogeo(frames)
  const anomala = tasaAnomala(phase.id, vs)

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <span className="corners" />
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/50 px-5 py-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          <Scramble text="Descenso 3D · corredor en vivo" />
        </h2>
        <div className="flex items-center gap-2">
          <span className="hidden font-mono text-[10.5px] text-muted-foreground/70 md:inline">
            {ok ? 'webgl · three.js' : 'modo compatibilidad'}
          </span>
          <Button
            variant="outline" size="sm"
            className={`h-7 gap-1.5 text-xs ${live ? 'border-[#ff4d5e]/50 text-[#ff8f9b] hover:bg-[#ff4d5e]/10' : ''}`}
            title="Seguir el último frame que llega del pipeline (vuelo real con GPS/IMU)"
            onClick={() => setLive(v => !v)}
            disabled={!frames.length}
          >
            <span className={`size-1.5 rounded-full ${live ? 'animate-pulse bg-[#ff4d5e]' : 'bg-[#6b7a8c]'}`} />
            {live ? 'En vivo' : 'Live'}
          </Button>
          <Button
            variant="outline" size="sm" className="h-7 gap-1.5 text-xs"
            title="Reproducir el descenso completo (selecciona frames en orden)"
            onClick={() => {
              if (!playing && frames.length) select(frames[0].src)
              setPlaying(p => !p)
            }}
            disabled={!frames.length}
          >
            {playing ? <Pause className="size-3" /> : <Play className="size-3" />}
            {playing ? 'Pausar' : 'Descenso'}
          </Button>
        </div>
      </div>

      <div ref={wrap} className="relative min-h-[360px] flex-1 md:min-h-[420px]">
        {ok ? (
          <Canvas
            frameloop={inView && !presenting && !jurado ? 'always' : 'never'}
            dpr={[1, 1.75]}
            gl={{ alpha: true, antialias: true, powerPreference: 'high-performance' }}
            camera={{ position: [8.4, 4.6, 11.2], fov: 42 }}
            className="!absolute !inset-0"
          >
            <SceneRig targetY={landed ? 0.8 : 1.5 + norm * 8.2} stowed={stowed} landed={landed} sol={sol} />
          </Canvas>
        ) : (
          <div className="absolute inset-0 p-6"><FallbackProfile /></div>
        )}

        {/* regla de altitud */}
        <div className="pointer-events-none absolute bottom-4 left-4 top-4 w-12">
          <div className="relative h-full border-l border-dashed border-[#3ddc84]/25">
            {ticks.map((t, i) => (
              <span
                key={i}
                className="absolute left-0 -translate-y-1/2 pl-2 font-mono text-[9.5px] text-muted-foreground/70"
                style={{ top: `${(1 - [1, 0.75, 0.5, 0.25, 0][i]) * 100}%` }}
              >
                {t} m
              </span>
            ))}
            <motion.span
              className="absolute left-[-4px] size-2 rounded-full bg-[#3ddc84] shadow-[0_0_10px_2px_rgba(61,220,132,.6)]"
              animate={{ top: `${(1 - norm) * 100}%` }}
              transition={{ type: 'spring', stiffness: 120, damping: 20 }}
            />
          </div>
        </div>

        {/* tape de altitud + variómetro */}
        <div className="pointer-events-none absolute right-4 top-4 flex flex-col items-end gap-1 font-mono">
          <span className="text-[9.5px] uppercase tracking-[0.14em] text-muted-foreground/70">alt · vs</span>
          <span className="text-[22px] font-semibold leading-none text-[#eafff3]">
            {num(alt, 1)}<span className="ml-0.5 text-[10px] text-muted-foreground">m</span>
          </span>
          <span className={`text-[12px] ${anomala ? 'text-[#ffb020]' : 'text-[#7ee8b0]'}`}>
            {vs === null ? '— m/s' : `${vs < 0 ? '▼' : '▲'} ${num(Math.abs(vs), 1)} m/s`}
          </span>
          <span className="text-[9.5px] text-muted-foreground/70">apogeo {apo ? num(apo.alt, 0) : '—'} m</span>
          <div className="relative mt-1 h-36 w-[3px] rounded-full bg-[#1c2530]">
            <motion.span
              className="absolute left-[-2.5px] size-2 rounded-full"
              style={{ background: phase.color }}
              animate={{ top: `${(1 - norm) * 100}%` }}
              transition={{ type: 'spring', stiffness: 120, damping: 20 }}
            />
          </div>
          {anomala && (
            <span className="mt-1 rounded-full border border-[#ffb020]/40 bg-[#ffb020]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-[#ffb020]">
              tasa anómala
            </span>
          )}
        </div>

        {/* HUD del frame seguido */}
        <div className="pointer-events-none absolute bottom-4 right-4 text-right font-mono text-[10.5px] leading-relaxed">
          <div className="flex items-center justify-end gap-2">
            {live && !playing && (
              <span className="flex items-center gap-1 rounded-full border border-[#ff4d5e]/40 bg-[#ff4d5e]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-[#ff8f9b]">
                <Radio className="size-2.5" /> en vivo
              </span>
            )}
            <motion.span key={focus?.src ?? 'none'} initial={{ opacity: 0.2 }} animate={{ opacity: 1 }} className="text-[13px] font-semibold text-[#eafff3]">
              {focus?.src ?? '—'}
            </motion.span>
          </div>
          <div className="text-muted-foreground">
            alt {num(alt, 1)} m · daño {num(focus?.danado_pct, 1)}%
          </div>
          <div className="text-muted-foreground/70">
            {playing ? 'reproduciendo descenso' : live ? 'sigue el último frame del pipeline' : 'el cansat sigue al frame seleccionado'}
          </div>
        </div>

        {!frames.length && (
          <div className="absolute inset-0 grid place-items-center">
            <span className="flex items-center gap-2 font-mono text-xs text-muted-foreground">
              <Box className="size-4" /> esperando telemetría del descenso…
            </span>
          </div>
        )}
      </div>
    </Card>
  )
}
