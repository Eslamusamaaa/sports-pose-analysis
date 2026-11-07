import { useRef, useState } from 'react'

function human(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(2)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

interface Props {
  busy: boolean
  uploadPct: number
  onAnalyze: (file: File) => void
  /** Disables analysing (not selecting) and says why, e.g. model not installed. */
  blockedReason?: string
}

export interface PoseEngine {
  id: string
  label: string
  keypoints: string
  license: string
  ready: boolean
}

export function UploadPanel({
  busy,
  uploadPct,
  onAnalyze,
  blockedReason,
}: Props) {
  const [file, setFile] = useState<File | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <section className="panel">
      <h2>Upload</h2>
      <div className="row">
        <input
          ref={inputRef}
          type="file"
          accept="video/*,.mp4,.avi,.mov,.mkv,.webm"
          style={{ display: 'none' }}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button onClick={() => inputRef.current?.click()} disabled={busy}>
          Select Video
        </button>
        <button
          className="primary"
          disabled={!file || busy || !!blockedReason}
          onClick={() => file && onAnalyze(file)}
        >
          {busy ? 'Processing…' : 'Analyze Video'}
        </button>
        {blockedReason && <span className="small muted">{blockedReason}</span>}
      </div>

      <div className="file-meta" style={{ marginTop: 12 }}>
        {file ? (
          <>
            selected: <strong>{file.name}</strong>
            <br />
            size: <strong>{human(file.size)}</strong>
            {file.type && (
              <>
                {' · '}type: <strong>{file.type}</strong>
              </>
            )}
          </>
        ) : (
          <span className="muted">No video selected.</span>
        )}
      </div>

      {busy && uploadPct > 0 && uploadPct < 100 && (
        <div style={{ marginTop: 12 }}>
          <div className="bar-track" style={{ height: 8 }}>
            <div className="bar-fill" style={{ width: `${uploadPct}%` }} />
          </div>
          <div className="small muted" style={{ marginTop: 4 }}>
            uploading {uploadPct}%
          </div>
        </div>
      )}
    </section>
  )
}
