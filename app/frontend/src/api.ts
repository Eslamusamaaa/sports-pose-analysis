import type {
  HealthResponse,
  JobStatusResponse,
  LogEntry,
  ResultResponse,
} from './types'

// Relative URLs: Vite proxies /api to the backend (see vite.config.ts).
const BASE = '/api'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export function getHealth(): Promise<HealthResponse> {
  return fetch(`${BASE}/health`).then(json<HealthResponse>)
}

export type Sport = 'karate' | 'swimming'

export function uploadVideo(
  file: File,
  onProgress?: (pct: number) => void,
  sport: Sport = 'karate',
  poseEngine = 'openpose',
): Promise<{ job_id: string; status: string }> {
  // XHR rather than fetch: fetch has no upload-progress events.
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)
    form.append('sport', sport)
    form.append('pose_engine', poseEngine)
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/analyze`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100))
    }
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText)
        if (xhr.status >= 200 && xhr.status < 300) resolve(body)
        else reject(new Error(body?.detail ?? `${xhr.status} ${xhr.statusText}`))
      } catch {
        reject(new Error(`Unexpected response (${xhr.status}): ${xhr.responseText.slice(0, 300)}`))
      }
    }
    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.send(form)
  })
}

export function getJob(jobId: string): Promise<JobStatusResponse> {
  return fetch(`${BASE}/jobs/${jobId}`).then(json<JobStatusResponse>)
}

export function getResult(jobId: string): Promise<ResultResponse> {
  return fetch(`${BASE}/jobs/${jobId}/result`).then(json<ResultResponse>)
}

export function getLogs(jobId: string): Promise<{ logs: LogEntry[] }> {
  return fetch(`${BASE}/jobs/${jobId}/logs`).then(json<{ logs: LogEntry[] }>)
}

export function getSkeletonSummary(jobId: string): Promise<unknown> {
  return fetch(`${BASE}/jobs/${jobId}/skeleton`).then(json<unknown>)
}

export const inputVideoUrl = (jobId: string) => `${BASE}/jobs/${jobId}/input-video`
export const openposeVideoUrl = (jobId: string) => `${BASE}/jobs/${jobId}/openpose-video`
export const skeletonJsonUrl = (jobId: string) => `${BASE}/jobs/${jobId}/skeleton?full=true`
export const predictionsUrl = (jobId: string) => `${BASE}/jobs/${jobId}/predictions`
