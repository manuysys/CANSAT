"""
CanSat La Base — Muestreo adaptativo (IIC + estratificado + incertidumbre).

────────────────────────────────────────────────────────────────────────────
CONTRATO DE ESCALA DE ``uncert`` (leer antes de tocar los umbrales)
────────────────────────────────────────────────────────────────────────────
``uncert`` SIEMPRE llega **normalizado a [0, 1]**, venga de donde venga:

  · ``mission_pipeline.entropy_uncertainty()`` → entropía del softmax dividida
    por ln(n_clases). Típico: 0.1 (frame limpio y seguro) a 0.7 (ambiguo).
  · ``mission_pipeline.tta_logits()`` → varianza entre las 4 vistas dividida
    por ``TTA_UNCERT_REF``. Típico: 0.05 a 0.4.

⚠ El bug que esto corrige: la varianza cruda de TTA (0.001-0.01) y la entropía
  normalizada (0.3-0.7) se comparaban contra el MISMO ``uncert_max=0.03``. Con
  la entropía (el camino sin ``--tta``, que es la configuración de vuelo) el
  score quedaba saturado (``unc=1.0`` siempre) y ``review=True`` casi siempre:
  el sampler mandaba todo a HIGH. Los dos caminos ahora están en la misma
  escala, así que ``--uncert-max`` aplica a ambos. El JSONL registra el valor
  crudo por frame: calibrar con el p95 de un vuelo de prueba.
"""
import numpy as np


class AdaptiveSampler:
    def __init__(self, w=(0.45, 0.25, 0.20, 0.10),
                 thr_high=0.30, thr_med=0.12, uncert_max=0.5):
        self.w = w
        self.thr_high, self.thr_med, self.uncert_max = thr_high, thr_med, uncert_max
        self.coverage = np.zeros(5)

    def interest_score(self, pcts, usi, uncert=0.0,
                       pct_dan=0.0, pct_flood=0.0):
        """pcts=[veg,bui,water,bare,other] en %. Devuelve 0..1."""
        w1, w2, w3, w4 = self.w
        dan = min(1.0, max(pct_dan, pct_flood) / 30.0)
        urban = min(1.0, max(0.0, usi) / 300.0)
        hetero = min(1.0, float(np.std(pcts)) / 50.0)
        # Escala atada al umbral de review: no pueden divergir otra vez.
        unc = min(1.0, max(0.0, uncert) / max(1e-6, self.uncert_max))
        return float(np.clip(w1 * dan + w2 * urban + w3 * hetero + w4 * unc,
                             0, 1))

    def coverage_boost(self, pcts):
        """Estratificado: premia frames con clases poco cubiertas."""
        frac = np.asarray(pcts, dtype=float) / 100.0
        self.coverage = 0.9 * self.coverage + 0.1 * frac
        boost = 0.0
        if self.coverage[1] < 0.05 and frac[1] > 0.2:
            boost += 0.15
        if self.coverage[2] < 0.05 and frac[2] > 0.2:
            boost += 0.15
        return boost

    def decide(self, score, uncert=0.0):
        review = uncert > self.uncert_max
        if score >= self.thr_high or review:
            return {"priority": "HIGH", "action": "high_res",
                    "review": review, "score": round(score, 3)}
        if score >= self.thr_med:
            return {"priority": "MEDIUM", "action": "full_res",
                    "review": False, "score": round(score, 3)}
        return {"priority": "LOW", "action": "thumb",
                "review": False, "score": round(score, 3)}
