import type { StageInfo } from '../types'

const LABELS: Record<string, string> = {
  upload: 'Video uploaded',
  validate: 'Video validated',
  openpose: 'OpenPose (BODY_25)',
  skeleton: 'Skeleton extracted',
  convert: 'MMAction2 input prepared (BODY_25 → COCO-17)',
  mmaction: 'MMAction2 inference',
  completed: 'Final result',
}

// The "openpose" stage runs whichever pose model the job was started with.
const ENGINE_LABELS: Record<string, string> = {
  openpose: 'OpenPose (BODY_25)',
  rtmpose: 'RTMPose-m + RTMDet-m (boxes + keypoints)',
  yolo11: 'YOLO11m-pose (boxes + keypoints + tracking)',
}

const CONVERT_LABELS: Record<string, string> = {
  openpose: 'Action-model input prepared (BODY_25 → COCO-17)',
  rtmpose: 'Action-model input prepared (Halpe26 → COCO-17)',
  yolo11: 'Action-model input prepared (COCO-17, native)',
}

function badge(state: string) {
  switch (state) {
    case 'done':
      return '✓'
    case 'running':
      return <span className="spin">⏳</span>
    case 'failed':
      return '✕'
    case 'skipped':
      return '–'
    default:
      return '○'
  }
}

export function PipelineStatus({
  stages,
  poseEngine,
  actionLabel,
}: {
  stages: StageInfo[]
  /** job.pose_engine - names the keypoint stage after the model really used */
  poseEngine?: string
  /** overrides the action-recognition stage label (e.g. the swimming model) */
  actionLabel?: string
}) {
  const label = (name: string) =>
    name === 'openpose'
      ? ENGINE_LABELS[poseEngine ?? 'openpose'] ?? LABELS.openpose
      : name === 'convert'
        ? CONVERT_LABELS[poseEngine ?? 'openpose'] ?? LABELS.convert
        : name === 'mmaction' && actionLabel
          ? actionLabel
          : name === 'skeleton'
            ? 'Tracking + players chosen'
            : LABELS[name] ?? name
  return (
    <section className="panel">
      <h2>Processing status</h2>
      <ul className="stages">
        {stages.map((s) => (
          <li key={s.name} className={`s-${s.state}`}>
            <span className="badge">{badge(s.state)}</span>
            <span>{label(s.name)}</span>
            {s.duration_s != null && <span className="stage-time">{s.duration_s}s</span>}
            {s.error && <span className="stage-time" style={{ color: 'var(--err)' }}>failed</span>}
          </li>
        ))}
      </ul>
    </section>
  )
}
