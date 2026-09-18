/* Silencia ruido de consola ajeno a la app: el deprecation interno de three.js
   (THREE.Clock, que usa @react-three/fiber por debajo) y los mensajes de
   performance del driver GL en entornos headless/SwiftShader. El resto de
   warnings y errors sigue intacto. */
const NOISE = [
  /THREE\.Clock: This module has been deprecated/,
  /GL Driver Message/,
  /GPU stall due to ReadPixels/,
]

const origWarn = console.warn.bind(console)
console.warn = (...args: unknown[]) => {
  const first = typeof args[0] === 'string' ? args[0] : ''
  if (NOISE.some(re => re.test(first))) return
  origWarn(...args)
}

export {}
