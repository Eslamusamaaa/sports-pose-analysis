import { useEffect, useRef, useState } from 'react'

// OpenPose BODY_25 bone pairs, from POSE_BODY_25_PAIRS in
// include/openpose/pose/poseParameters.hpp.
const BODY_25_PAIRS: [number, number][] = [
  [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
  [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
  [1, 0], [0, 15], [15, 17], [0, 16], [16, 18],
  [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24],
]

// Per-person colours so two fighters stay visually distinct.
const PERSON_COLOURS = ['#4da3ff', '#3ddc97', '#ffb74d', '#ff6b6b']

export type Keypoint = [number, number, number] // x, y, confidence

/** A labelled player from the backend (Player 1 / Player 2 ...). */
export interface LivePlayer {
  label: string
  slot: number
  keypoints: Keypoint[]
  bbox?: [number, number, number, number] | null
}

// Fixed colour per player slot: Player 1 blue, Player 2 orange (matches the
// batch overlay video).
const SLOT_COLOURS = ['#28a0ff', '#ff9628', '#5adc5a', '#dc5adc']

interface Props {
  /** When given, ONLY these players are drawn (box + skeleton + label). */
  players?: LivePlayer[] | null
  people: Keypoint[][] | null
  frameHw: [number, number] | null // [height, width] of the camera frame
  /** base64 JPEG of the SAME frame, drawn underneath the skeleton. */
  background?: string | null
  minConfidence?: number
  mirror?: boolean
}

// Same person in both lists? (players are a subset of everyone detected)
function sameSkeleton(a: Keypoint[], b: Keypoint[]): boolean {
  if (!a || !b || a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i][2] > 0 && (Math.abs(a[i][0] - b[i][0]) > 1 || Math.abs(a[i][1] - b[i][1]) > 1)) return false
  }
  return true
}

export function SkeletonCanvas({
  players = null,
  people,
  frameHw,
  background = null,
  minConfidence = 0.15,
  mirror = true,
}: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  // Bumped when a decoded frame is ready, to re-run the draw effect. The image
  // decodes asynchronously, so drawing it must wait for onload or the canvas
  // would paint an empty bitmap.
  const [imgTick, setImgTick] = useState(0)

  useEffect(() => {
    if (!background) {
      imgRef.current = null
      return
    }
    let cancelled = false
    const img = new Image()
    img.onload = () => {
      if (cancelled) return
      imgRef.current = img
      setImgTick((t) => t + 1)
    }
    img.src = `data:image/jpeg;base64,${background}`
    return () => {
      cancelled = true
    }
  }, [background])

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const [srcH, srcW] = frameHw ?? [480, 640]
    // Render at the camera's aspect ratio, sized to the element's box.
    const cssW = canvas.clientWidth || 640
    const cssH = Math.round((cssW * srcH) / srcW)
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr)
      canvas.height = Math.round(cssH * dpr)
      canvas.style.height = `${cssH}px`
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cssW, cssH)
    ctx.fillStyle = '#0b0e12'
    ctx.fillRect(0, 0, cssW, cssH)

    // Draw the camera frame underneath. It is the SAME frame the keypoints came
    // from and is stretched to the same box, so the skeleton lines up exactly.
    const bg = imgRef.current
    if (bg) {
      ctx.save()
      if (mirror) {
        ctx.translate(cssW, 0)
        ctx.scale(-1, 1)
      }
      ctx.drawImage(bg, 0, 0, cssW, cssH)
      ctx.restore()
      // Darken slightly so the skeleton stays readable over a bright scene.
      ctx.fillStyle = 'rgba(0,0,0,0.28)'
      ctx.fillRect(0, 0, cssW, cssH)
    }

    type Drawn = { kps: Keypoint[]; colour: string; label?: string; bbox?: number[] | null; thin?: boolean }
    // With players: everyone detected is drawn thin grey (box from the joint
    // extent), then the players on top in their colour with "Player N".
    const drawn: Drawn[] = players
      ? [
          ...(people ?? [])
            .filter((kps) => !players.some((p) => sameSkeleton(p.keypoints, kps)))
            .map((kps) => ({ kps, colour: '#9aa4b2', bbox: null, label: undefined, thin: true })),
          ...players.map((p) => ({
            kps: p.keypoints,
            colour: SLOT_COLOURS[p.slot % SLOT_COLOURS.length],
            label: p.label,
            bbox: p.bbox,
          })),
        ]
      : (people ?? []).map((kps, pi) => ({ kps, colour: PERSON_COLOURS[pi % PERSON_COLOURS.length] }))

    if (drawn.length === 0) {
      ctx.fillStyle = bg ? '#c8d0da' : '#6b7684'
      ctx.font = '13px system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(players ? 'no player detected' : 'no person detected', cssW / 2, cssH / 2)
      return
    }

    const sx = cssW / srcW
    const sy = cssH / srcH
    // Mirror so the view behaves like a mirror, matching the webcam preview.
    const px = (x: number) => (mirror ? cssW - x * sx : x * sx)
    const py = (y: number) => y * sy

    drawn.forEach(({ kps, colour, label, bbox, thin }) => {
      if (!kps || kps.length < 25) return
      // Box: the detector's (RTMDet / YOLO11) or, for OpenPose, the joint extent.
      let box = bbox ?? null
      if (!box && (label || thin)) {
        const pts = kps.filter((k) => k[2] >= minConfidence)
        if (pts.length >= 2) {
          const xs = pts.map((k) => k[0])
          const ys = pts.map((k) => k[1])
          box = [Math.min(...xs) - 12, Math.min(...ys) - 12, Math.max(...xs) + 12, Math.max(...ys) + 12]
        }
      }
      if (box) {
        const [x1, y1, x2, y2] = box
        const left = Math.min(px(x1), px(x2))
        const top = py(y1)
        ctx.lineWidth = thin ? 1 : 2
        ctx.strokeStyle = colour
        ctx.strokeRect(left, top, Math.abs(px(x2) - px(x1)), py(y2) - py(y1))
        if (label) {
          ctx.font = 'bold 13px system-ui, sans-serif'
          ctx.textAlign = 'left'
          const tw = ctx.measureText(label).width
          const ty = top - 18 > 0 ? top - 18 : top
          ctx.fillStyle = colour
          ctx.fillRect(left, ty, tw + 10, 18)
          ctx.fillStyle = '#ffffff'
          ctx.fillText(label, left + 5, ty + 13)
        }
      }

      ctx.lineWidth = thin ? 1.5 : 3
      ctx.strokeStyle = colour
      ctx.lineCap = 'round'
      for (const [a, b] of BODY_25_PAIRS) {
        const ka = kps[a]
        const kb = kps[b]
        if (!ka || !kb) continue
        // (0,0,0) means "not detected" in OpenPose output.
        if (ka[2] < minConfidence || kb[2] < minConfidence) continue
        ctx.beginPath()
        ctx.moveTo(px(ka[0]), py(ka[1]))
        ctx.lineTo(px(kb[0]), py(kb[1]))
        ctx.stroke()
      }

      ctx.fillStyle = colour
      for (const k of kps) {
        if (!k || k[2] < minConfidence) continue
        ctx.beginPath()
        ctx.arc(px(k[0]), py(k[1]), 3.2, 0, Math.PI * 2)
        ctx.fill()
      }
    })
  }, [players, people, frameHw, minConfidence, mirror, imgTick])

  return <canvas ref={ref} style={{ width: '100%', borderRadius: 8, display: 'block' }} />
}
