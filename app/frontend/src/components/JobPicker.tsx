import { useCallback, useEffect, useState } from 'react'

interface JobSummary {
  job_id: string
  filename: string
  status: string
  stage: string
  created_at: string
}

const STATUS_ICON: Record<string, string> = {
  completed: '✓',
  partial: '◐',
  failed: '✕',
  running: '⏳',
  queued: '○',
}

/**
 * Lists jobs held by the backend so a run started elsewhere (another tab, a
 * curl call, or before a page reload) can still be opened. Without this the UI
 * can only ever show a job it started itself in the current session.
 */
export function JobPicker({
  currentJobId,
  onOpen,
  sport,
}: {
  currentJobId: string | null
  onOpen: (jobId: string) => void
  /** Only list jobs of this sport, so each page shows its own history. */
  sport?: string
}) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [open, setOpen] = useState(false)

  const refresh = useCallback(() => {
    fetch(sport ? `/api/jobs?sport=${sport}` : '/api/jobs')
      .then((r) => r.json())
      .then((d) => setJobs(d.jobs ?? []))
      .catch(() => undefined)
  }, [sport])

  useEffect(() => {
    refresh()
    // Cheap poll so a running job's status stays current in the list.
    const t = window.setInterval(refresh, 5000)
    return () => window.clearInterval(t)
  }, [refresh])

  if (jobs.length === 0) return null

  return (
    <section className="panel">
      <h2>
        Previous analyses ({jobs.length})
      </h2>
      <div className="row" style={{ marginBottom: open ? 12 : 0 }}>
        <button onClick={() => setOpen((o) => !o)}>
          {open ? 'Hide' : 'Show'} job list
        </button>
        <span className="small muted">
          Open a finished run without re-uploading the video.
        </span>
      </div>

      {open && (
        <table className="kv">
          <thead>
            <tr>
              <th>Status</th>
              <th>File</th>
              <th>Job id</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.job_id}>
                <td>
                  {STATUS_ICON[j.status] ?? '•'} {j.status}
                  {j.status === 'running' && (
                    <span className="muted small"> ({j.stage})</span>
                  )}
                </td>
                <td>{j.filename}</td>
                <td className="mono small">{j.job_id}</td>
                <td>
                  <button
                    onClick={() => onOpen(j.job_id)}
                    disabled={j.job_id === currentJobId}
                  >
                    {j.job_id === currentJobId ? 'open' : 'View'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}
