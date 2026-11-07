import type { LogEntry, OutputEntry } from '../types'

// The pose stage is stored under the internal key "openpose" whatever model
// ran it; show the model that really did the work.
const POSE_STAGE_TAG: Record<string, string> = { openpose: 'openpose', rtmpose: 'rtmpose', yolo11: 'yolo11' }

function human(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`
}

export function LogPanel({
  logs,
  outputs,
  poseEngine,
}: {
  logs: LogEntry[]
  outputs: Record<string, OutputEntry> | null
  poseEngine?: string
}) {
  const tag = (stage: string) =>
    stage === 'openpose' ? POSE_STAGE_TAG[poseEngine ?? 'openpose'] ?? 'pose' : stage
  return (
    <section className="panel">
      <h2>Logs &amp; output files</h2>

      {outputs && (
        <table className="kv" style={{ marginBottom: 16 }}>
          <thead>
            <tr>
              <th>Artefact</th>
              <th>Size</th>
              <th>Path</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(outputs).map(([name, o]) => (
              <tr key={name}>
                <td>
                  {o.exists ? '✓' : '○'} {name}
                  {o.file_count != null && <span className="muted"> ({o.file_count} files)</span>}
                </td>
                <td className="mono">{o.exists ? human(o.size_bytes) : '—'}</td>
                <td className="mono small" style={{ wordBreak: 'break-all' }}>
                  {o.path}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div className="logs">
        {logs.length === 0 ? (
          <div className="muted">No log entries yet.</div>
        ) : (
          logs.map((l, i) => (
            <div key={i} className={`lg-${l.level}`}>
              {l.ts.slice(11, 23)} {l.level.toUpperCase().padEnd(7)}
              {l.stage ? `[${tag(l.stage)}] ` : ''}
              {l.message}
            </div>
          ))
        )}
      </div>
    </section>
  )
}
