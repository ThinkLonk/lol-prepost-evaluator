export class ApiError extends Error {
  constructor(public code: string, message: string) { super(message) }
}

export async function api<T>(path: string, method = 'GET', body?: unknown, session?: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, {
      method,
      headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...(session ? { 'X-Analysis-Session': session } : {}) },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: 'no-store',
    })
  } catch {
    throw new ApiError('E_NETWORK', method === 'POST' && ['/pre', '/post'].includes(path)
      ? 'Mất kết nối với máy chủ. Kết quả có thể đã được lưu; hãy thử lại cùng yêu cầu.'
      : 'Mất kết nối với máy chủ. Kiểm tra ứng dụng FastAPI và thử lại.')
  }
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(data?.code ?? 'E_HTTP', data?.message ?? 'Không xử lý được yêu cầu. Hãy thử lại.')
  return data as T
}
